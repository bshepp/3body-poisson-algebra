#!/bin/bash
# Userdata for GUE log-gas Poisson algebra computation on EC2.
# Installs deps, pulls code from S3, runs the comparison, syncs results back.

LOG=/var/log/3body-gue-logas.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# ---- CONFIGURABLE: set these before launching ----
GUE_ARGS="__GUE_ARGS__"
JOB_NAME="__JOB_NAME__"
# ---------------------------------------------------

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== GUE Log-Gas Computation ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "Job:      $JOB_NAME"
echo "Args:     $GUE_ARGS"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/primes/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"$JOB_NAME\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/results/primes/instance_heartbeat.json" 2>/dev/null || true
}

MAIN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Received at $(date -u)"
    if [ -n "$MAIN_PID" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
        kill -TERM "$MAIN_PID"
        for i in $(seq 1 60); do
            if ! kill -0 "$MAIN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    # Sync results and checkpoints
    aws s3 sync "$WORKDIR/primes/results/" "s3://$S3_BUCKET/results/primes/" \
        --exclude "*.pyc" 2>/dev/null || true
    upload_log
    exit 0
}
trap shutdown_handler SIGTERM

echo "=== Step 1: Installing dependencies ==="
upload_log

yum install -y python3 python3-pip python3-devel gcc gcc-c++
amazon-linux-extras install python3.8 -y 2>/dev/null || true
PYTHON=$(command -v python3.8 || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip 2>/dev/null || true
$PYTHON -m pip install sympy==1.13.3 numpy scipy mpmath

echo "=== Step 2: Checking versions ==="
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import numpy; print('NumPy:', numpy.__version__)"
$PYTHON -c "import scipy; print('SciPy:', scipy.__version__)"
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR/primes" "$WORKDIR/nbody"
aws s3 sync "s3://$S3_BUCKET/code/nbody/" "$WORKDIR/nbody/"
aws s3 sync "s3://$S3_BUCKET/code/primes/" "$WORKDIR/primes/"

echo "Key files:"
ls -la "$WORKDIR/nbody/exact_growth_nbody.py" 2>/dev/null || echo "exact_growth_nbody.py NOT FOUND"
ls -la "$WORKDIR/primes/run_gue_logas.py" 2>/dev/null || echo "run_gue_logas.py NOT FOUND"
upload_log

echo "=== Step 4: Pulling any existing checkpoints ==="
aws s3 sync "s3://$S3_BUCKET/results/primes/" "$WORKDIR/primes/results/" \
    --exclude "*.log" --exclude "*.json" \
    --include "checkpoints_*/**" 2>/dev/null || echo "(no prior checkpoints)"
upload_log

echo "=== Step 5: Starting background sync ==="
(while true; do
    sleep 120
    aws s3 sync "$WORKDIR/primes/results/" "s3://$S3_BUCKET/results/primes/" \
        --exclude "*.pyc" 2>/dev/null || true
    upload_log
done) &
LOG_SYNC_PID=$!

echo "=== Step 6: Running GUE log-gas computation ==="
echo "Command: $PYTHON -u run_gue_logas.py $GUE_ARGS"
echo "=== $(date -u) ==="

cd "$WORKDIR/primes"
$PYTHON -u run_gue_logas.py $GUE_ARGS --resume &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
aws s3 sync "$WORKDIR/primes/results/" "s3://$S3_BUCKET/results/primes/" \
    --exclude "*.pyc" || true

upload_log
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/primes/full.log" || true

cat > "/tmp/gue_completion.json" <<CEOF
{
    "status": "complete",
    "job": "$JOB_NAME",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/gue_completion.json" \
    "s3://$S3_BUCKET/results/primes/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

shutdown -h now 2>/dev/null || true
