#!/bin/bash
# ==========================================================================
# AWS Spot Instance Launcher -- Composite Potentials & Post-Newtonian
#
# Runs three phases:
#   1. Composite universality test: V = -u - u^2, V = -u - u^2/2 - 3u^3/10
#   2. Static 1PN three-body: c=10, 50, 100 + Newtonian baseline
#   3. 1PN mass invariance: equal, hierarchical, mixed masses
#
# All runs through level 3 (dimension 116 test).
# ==========================================================================

LOG=/var/log/3body-composite-pn.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== Composite Potentials & Post-Newtonian ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "CPUs: $(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())' 2>/dev/null || echo unknown)"

# ---------- Trap SIGTERM for graceful shutdown ----------------------------
SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation signal received at $(date -u)"
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        echo "[SIGTERM] Forwarding SIGTERM to run_pn_aws.py PID $SCAN_PID"
        kill -TERM "$SCAN_PID"
        for i in $(seq 1 18); do
            if ! kill -0 "$SCAN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    echo "[SIGTERM] Emergency sync..."
    cd "$WORKDIR/nbody"
    for d in checkpoints_*; do
        [ -d "$d" ] && aws s3 sync "$d" "s3://$S3_BUCKET/nbody_checkpoints/$d" || true
    done
    aws s3 cp "$WORKDIR/nbody/pn_completion.json" \
        "s3://$S3_BUCKET/results/composite_pn/pn_completion.json" || true
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/composite_pn/full.log" || true
    echo "[SIGTERM] Shutdown complete at $(date -u)"
    exit 0
}
trap shutdown_handler SIGTERM

# ---------- Install dependencies (Python 3.8+ for SymPy 1.13.3) -----------
dnf install -y python3 python3-pip 2>/dev/null || yum install -y python3 python3-pip
amazon-linux-extras install python3.8 -y 2>/dev/null || true
PYTHON=$(command -v python3.8 || command -v python3)
$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip 2>/dev/null || true
$PYTHON -m pip install sympy numpy scipy matplotlib

echo "Python: $($PYTHON --version)"
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
echo "NumPy:  $($PYTHON -c 'import numpy; print(numpy.__version__)')"

# ---------- Pull code from S3 --------------------------------------------
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Pulling code ==="
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "checkpoints*"

echo "Code files in nbody/:"
ls -la "$WORKDIR/nbody/"

# ---------- Pull existing checkpoints (for resume) -----------------------
echo "=== Pulling existing checkpoints for resume ==="
cd "$WORKDIR/nbody"
aws s3 sync "s3://$S3_BUCKET/nbody_checkpoints/" "$WORKDIR/nbody/" \
    --include "checkpoints_*/*" || true

aws s3 cp "s3://$S3_BUCKET/results/composite_pn/pn_completion.json" \
    "$WORKDIR/nbody/pn_completion.json" 2>/dev/null || true

echo "Checkpoint dirs:"
ls -d checkpoints_* 2>/dev/null || echo "  (none)"

if [ -f pn_completion.json ]; then
    echo "Completion manifest:"
    cat pn_completion.json
fi

# ---------- Background log sync (every 60s) -------------------------------
sync_logs() {
    while true; do
        sleep 60
        aws s3 cp "$LOG" \
            "s3://$S3_BUCKET/results/composite_pn/live.log" 2>/dev/null || true
        echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"composite_pn\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
            | aws s3 cp - "s3://$S3_BUCKET/results/composite_pn/heartbeat.json" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# ---------- Run the orchestrator ------------------------------------------
echo ""
echo "=================================================================="
echo "=== Starting composite/PN orchestrator ==="
echo "=== $(date -u) ==="
echo "=================================================================="

cd "$WORKDIR/nbody"
$PYTHON -u run_pn_aws.py --max-level 3 --samples 500 &
SCAN_PID=$!
wait $SCAN_PID
EXIT_CODE=$?
SCAN_PID=""

echo "=== Orchestrator exit code: $EXIT_CODE ==="

# ---------- Final sync (everything, with retries) -------------------------
echo "=== Final data sync ==="
cd "$WORKDIR/nbody"
for d in checkpoints_*; do
    [ -d "$d" ] && aws s3 sync "$d" "s3://$S3_BUCKET/nbody_checkpoints/$d" || true
done

for attempt in 1 2 3; do
    echo "  Sync attempt $attempt..."
    aws s3 cp pn_completion.json \
        "s3://$S3_BUCKET/results/composite_pn/pn_completion.json" && break
    sleep 10
done

aws s3 cp pn_final_summary.json \
    "s3://$S3_BUCKET/results/composite_pn/pn_final_summary.json" 2>/dev/null || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/composite_pn/full.log" || true

# ---------- Write completion marker ---------------------------------------
cat > "$WORKDIR/nbody/aws_completion.json" <<CEOF
{
    "status": "complete",
    "job": "composite_pn",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "$WORKDIR/nbody/aws_completion.json" \
    "s3://$S3_BUCKET/results/composite_pn/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null

echo "=== ALL DONE: $(date -u) ==="
echo "=== Exit code: $EXIT_CODE ==="
