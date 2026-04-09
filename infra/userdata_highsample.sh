#!/bin/bash
# ==========================================================================
# AWS Instance -- Level 4 High-Sample Refinement
#
# Pushes d(4) beyond the current 4,501 lower bound using 50K-200K samples.
# Fixed early-stop logic: uses boundary gap, not noise-tail gap.
# ==========================================================================

LOG=/var/log/3body-level4-refine.log
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
echo "=== Level 4 High-Sample Refinement ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "CPUs: $(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())' 2>/dev/null || echo unknown)"
echo "RAM:  $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo unknown)"

# ---------- Trap SIGTERM for graceful shutdown ----------------------------
SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation signal received at $(date -u)"
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        echo "[SIGTERM] Forwarding to PID $SCAN_PID"
        kill -TERM "$SCAN_PID"
        for i in $(seq 1 18); do
            if ! kill -0 "$SCAN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    echo "[SIGTERM] Emergency sync..."
    cd "$WORKDIR"
    aws s3 sync results/ "s3://$S3_BUCKET/results/" || true
    aws s3 sync checkpoints/ "s3://$S3_BUCKET/checkpoints/" || true
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/level4_refine/full.log" || true
    echo "[SIGTERM] Shutdown complete at $(date -u)"
    exit 0
}
trap shutdown_handler SIGTERM

# ---------- Install dependencies (Python 3.8+ for SymPy 1.13.3) -----------
dnf install -y python3 python3-pip python3-devel gcc 2>/dev/null || \
    yum install -y python3 python3-pip python3-devel gcc
amazon-linux-extras install python3.8 -y 2>/dev/null || true
PYTHON=$(command -v python3.8 || command -v python3)
$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip 2>/dev/null || true
$PYTHON -m pip install sympy numpy scipy matplotlib

echo "Python: $($PYTHON --version)"
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
echo "NumPy:  $($PYTHON -c 'import numpy; print(numpy.__version__)')"

# ---------- Pull code from S3 --------------------------------------------
mkdir -p "$WORKDIR/checkpoints" "$WORKDIR/results"
cd "$WORKDIR"

echo "=== Pulling code ==="
aws s3 cp "s3://$S3_BUCKET/code/exact_growth.py" "$WORKDIR/exact_growth.py"
aws s3 cp "s3://$S3_BUCKET/code/level4_highsample.py" "$WORKDIR/level4_highsample.py"

# ---------- Pull Level 3 checkpoint and any existing L4 results -----------
echo "=== Pulling checkpoints ==="
aws s3 sync "s3://$S3_BUCKET/checkpoints/" "$WORKDIR/checkpoints/"

echo "=== Pulling existing L4 results (for resume) ==="
aws s3 sync "s3://$S3_BUCKET/results/" "$WORKDIR/results/" \
    --include "level4_global_*/*" --include "highsample_status.json" || true

echo "Checkpoint files:"
ls -la "$WORKDIR/checkpoints/"

# ---------- Background log sync (every 60s) -------------------------------
sync_logs() {
    while true; do
        sleep 60
        aws s3 cp "$LOG" \
            "s3://$S3_BUCKET/results/level4_refine/live.log" 2>/dev/null || true
        echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"level4_refine\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
            | aws s3 cp - "s3://$S3_BUCKET/results/level4_refine/heartbeat.json" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# ---------- Run the Level 4 script ----------------------------------------
echo ""
echo "=================================================================="
echo "=== Starting Level 4 high-sample refinement ==="
echo "=== $(date -u) ==="
echo "=================================================================="

cd "$WORKDIR"
$PYTHON -u level4_highsample.py &
SCAN_PID=$!
wait $SCAN_PID
EXIT_CODE=$?
SCAN_PID=""

echo "=== Script exit code: $EXIT_CODE ==="

# ---------- Final sync ----------------------------------------------------
echo "=== Final data sync ==="
cd "$WORKDIR"
aws s3 sync results/ "s3://$S3_BUCKET/results/" || true
aws s3 sync checkpoints/ "s3://$S3_BUCKET/checkpoints/" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/level4_refine/full.log" || true

cat > "$WORKDIR/aws_level4_refine_completion.json" <<CEOF
{
    "status": "complete",
    "job": "level4_refine",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "$WORKDIR/aws_level4_refine_completion.json" \
    "s3://$S3_BUCKET/results/level4_refine/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null

echo "=== ALL DONE: $(date -u) ==="
echo "=== Exit code: $EXIT_CODE ==="
