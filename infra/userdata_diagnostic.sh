#!/bin/bash
# ==========================================================================
# AWS Instance -- NBodyAlgebra Diagnostic
# Loads level-3 checkpoints from both engines, verifies symbolic identity,
# runs full SVD with parallel subs() evaluation.
# ==========================================================================

LOG=/var/log/3body-diagnostic.log
exec > "$LOG" 2>&1
set -x
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body-diagnostic"
export WORKDIR PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== NBodyAlgebra Diagnostic ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "CPUs: $(nproc)"
echo "RAM:  $(free -h | grep Mem | awk '{print $2}')"

# ---------- Install dependencies ------------------------------------------
dnf install -y python3 python3-pip 2>/dev/null || yum install -y python3 python3-pip
pip3 install sympy numpy

echo "Python: $(python3 --version)"
python3 -c "import sympy; print('SymPy:', sympy.__version__)"

# ---------- Pull code and checkpoints ------------------------------------
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Pulling diagnostic script ==="
aws s3 cp "s3://$S3_BUCKET/code/diagnostic_aws.py" "$WORKDIR/diagnostic_aws.py"

echo "=== Pulling checkpoints ==="
mkdir -p "$WORKDIR/checkpoints_exact_growth"
mkdir -p "$WORKDIR/checkpoints_NBodyAlgebra"

aws s3 sync "s3://$S3_BUCKET/diagnostic/checkpoints_exact_growth/" \
    "$WORKDIR/checkpoints_exact_growth/"
aws s3 sync "s3://$S3_BUCKET/diagnostic/checkpoints_NBodyAlgebra/" \
    "$WORKDIR/checkpoints_NBodyAlgebra/"

echo "Checkpoint files:"
ls -la "$WORKDIR/checkpoints_exact_growth/"
ls -la "$WORKDIR/checkpoints_NBodyAlgebra/"

# ---------- Background log sync ------------------------------------------
sync_logs() {
    while true; do
        sleep 30
        aws s3 cp "$LOG" "s3://$S3_BUCKET/diagnostic/live.log" 2>/dev/null || true
        echo "{\"instance\": \"$INSTANCE_ID\", \"job\": \"diagnostic\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
            | aws s3 cp - "s3://$S3_BUCKET/diagnostic/heartbeat.json" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# ---------- Run diagnostic -----------------------------------------------
echo ""
echo "=================================================================="
echo "=== Starting diagnostic ==="
echo "=== $(date -u) ==="
echo "=================================================================="

cd "$WORKDIR"
python3 -u diagnostic_aws.py
EXIT_CODE=$?

echo "=== Diagnostic exit code: $EXIT_CODE ==="
echo "=== Completed: $(date -u) ==="

# ---------- Final sync ---------------------------------------------------
aws s3 cp "$LOG" "s3://$S3_BUCKET/diagnostic/full.log" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== ALL DONE ==="
