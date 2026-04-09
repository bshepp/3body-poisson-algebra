#!/bin/bash
# Userdata for N-body scaling probe on EC2.
# Installs deps, pulls code from S3, runs the probe, syncs results back.

LOG=/var/log/3body-nbody-scaling.log
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
echo "=== N-body Scaling Probe ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/nbody_scaling/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"nbody_scaling\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/results/nbody_scaling/instance_heartbeat.json" 2>/dev/null || true
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
    aws s3 sync "$WORKDIR/nbody/" "s3://$S3_BUCKET/results/nbody_scaling/" \
        --include "*.json" --exclude "*.pyc" 2>/dev/null || true
    # Sync checkpoints
    for d in "$WORKDIR"/nbody/checkpoints_N*; do
        [ -d "$d" ] && aws s3 sync "$d" "s3://$S3_BUCKET/results/nbody_scaling/checkpoints/$(basename $d)/" 2>/dev/null || true
    done
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
mkdir -p "$WORKDIR/nbody"
aws s3 sync "s3://$S3_BUCKET/code/nbody/" "$WORKDIR/nbody/"

echo "Key files:"
ls -la "$WORKDIR/nbody/exact_growth_nbody.py" 2>/dev/null || echo "exact_growth_nbody.py NOT FOUND"
ls -la "$WORKDIR/nbody/run_n_scaling_aws.py" 2>/dev/null || echo "run_n_scaling_aws.py NOT FOUND"
upload_log

echo "=== Step 4: Pulling any existing checkpoints ==="
aws s3 sync "s3://$S3_BUCKET/results/nbody_scaling/checkpoints/" "$WORKDIR/nbody/" \
    --exclude "*" --include "checkpoints_N*/**" 2>/dev/null || echo "(no prior checkpoints)"
upload_log

echo "=== Step 5: Starting background log sync ==="
(while true; do
    sleep 120
    upload_log
done) &
LOG_SYNC_PID=$!

echo "=== Step 6: Running N-body scaling probe ==="
echo "Command: $PYTHON -u run_n_scaling_aws.py __SCALING_ARGS__"
echo "=== $(date -u) ==="

cd "$WORKDIR/nbody"
$PYTHON -u run_n_scaling_aws.py __SCALING_ARGS__ &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
aws s3 sync "$WORKDIR/nbody/" "s3://$S3_BUCKET/results/nbody_scaling/" \
    --include "*.json" --exclude "*.pyc" || true
for d in "$WORKDIR"/nbody/checkpoints_N*; do
    [ -d "$d" ] && aws s3 sync "$d" "s3://$S3_BUCKET/results/nbody_scaling/checkpoints/$(basename $d)/" || true
done

upload_log
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/nbody_scaling/full.log" || true

cat > "/tmp/nbody_scaling_completion.json" <<CEOF
{
    "status": "complete",
    "job": "nbody_scaling",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/nbody_scaling_completion.json" \
    "s3://$S3_BUCKET/results/nbody_scaling/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

shutdown -h now 2>/dev/null || true
