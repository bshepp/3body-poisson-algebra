#!/bin/bash
# ==========================================================================
# AWS Spot Instance -- Multi-System Universality Survey (Dimension Sequences)
#
# Runs NBodyAlgebra.compute_growth for all 21 scenarios defined in
# expansion_configs.py, testing the universality conjecture [3,6,17,116].
# ==========================================================================

LOG=/var/log/3body-expansion-dimseq.log
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
echo "=== Multi-System Universality Survey -- Dimension Sequences ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "CPUs: $(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())' 2>/dev/null || echo unknown)"
echo "RAM:  $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo unknown)"

# ---------- Trap SIGTERM for graceful shutdown ----------------------------
SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation signal received at $(date -u)"
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        echo "[SIGTERM] Forwarding to orchestrator PID $SCAN_PID"
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
    aws s3 cp "$WORKDIR/nbody/expansion_dimseq_completion.json" \
        "s3://$S3_BUCKET/results/expansion_dimseq/expansion_dimseq_completion.json" || true
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/expansion_dimseq/full.log" || true
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

aws s3 cp "s3://$S3_BUCKET/results/expansion_dimseq/expansion_dimseq_completion.json" \
    "$WORKDIR/nbody/expansion_dimseq_completion.json" 2>/dev/null || true

echo "Checkpoint dirs:"
ls -d checkpoints_* 2>/dev/null || echo "  (none)"

if [ -f expansion_dimseq_completion.json ]; then
    echo "Completion manifest:"
    cat expansion_dimseq_completion.json
fi

# ---------- Background log sync (every 60s) -------------------------------
sync_logs() {
    while true; do
        sleep 60
        aws s3 cp "$LOG" \
            "s3://$S3_BUCKET/results/expansion_dimseq/live.log" 2>/dev/null || true
        echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"expansion_dimseq\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
            | aws s3 cp - "s3://$S3_BUCKET/results/expansion_dimseq/heartbeat.json" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# ---------- Run the orchestrator ------------------------------------------
echo ""
echo "=================================================================="
echo "=== Starting expansion dimension sequence orchestrator ==="
echo "=== $(date -u) ==="
echo "=================================================================="

cd "$WORKDIR/nbody"
$PYTHON -u run_expansion_dimseq.py --max-level 3 --samples 500 &
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
    aws s3 cp expansion_dimseq_completion.json \
        "s3://$S3_BUCKET/results/expansion_dimseq/expansion_dimseq_completion.json" && break
    sleep 10
done

aws s3 cp expansion_dimseq_summary.json \
    "s3://$S3_BUCKET/results/expansion_dimseq/expansion_dimseq_summary.json" 2>/dev/null || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/expansion_dimseq/full.log" || true

# ---------- Write completion marker ---------------------------------------
cat > "$WORKDIR/nbody/aws_expansion_dimseq_completion.json" <<CEOF
{
    "status": "complete",
    "job": "expansion_dimseq",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "$WORKDIR/nbody/aws_expansion_dimseq_completion.json" \
    "s3://$S3_BUCKET/results/expansion_dimseq/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null

echo "=== ALL DONE: $(date -u) ==="
echo "=== Exit code: $EXIT_CODE ==="
