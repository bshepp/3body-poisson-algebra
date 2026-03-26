#!/bin/bash
LOG=/var/log/3body-bell-test.log
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
echo "=== Bell Test: CHSH Computation ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

CKPT_DIR="$WORKDIR/results/bell_test"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/bell_test/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"bell_test\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/results/bell_test/heartbeat.json" 2>/dev/null || true
}

MAIN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Received at $(date -u)"
    if [ -n "$MAIN_PID" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
        kill -TERM "$MAIN_PID"
        for i in $(seq 1 24); do
            if ! kill -0 "$MAIN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    aws s3 sync "$CKPT_DIR/" "s3://$S3_BUCKET/results/bell_test/" 2>/dev/null || true
    aws s3 sync "$WORKDIR/nbody/bell_test_results/" "s3://$S3_BUCKET/results/bell_test/plots/" 2>/dev/null || true
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
$PYTHON -m pip install sympy==1.13.3 numpy scipy matplotlib

echo "=== Step 2: Checking versions ==="
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import numpy; print('NumPy:', numpy.__version__)"
$PYTHON -c "import scipy; print('SciPy:', scipy.__version__)"
$PYTHON -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR"
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "checkpoints*"

echo "Key files:"
ls -la "$WORKDIR/nbody/bell_test.py" 2>/dev/null || echo "bell_test.py NOT FOUND"
ls -la "$WORKDIR/exact_growth.py" 2>/dev/null || echo "exact_growth.py NOT FOUND"
upload_log

echo "=== Step 4: Pulling Level 3 checkpoint ==="
mkdir -p "$WORKDIR/checkpoints"
aws s3 cp "s3://$S3_BUCKET/checkpoints/level_3.pkl" "$WORKDIR/checkpoints/level_3.pkl"
ls -la "$WORKDIR/checkpoints/level_3.pkl" || { echo "FATAL: level_3.pkl not found"; upload_log; exit 1; }

echo "=== Step 5: Pulling any prior bell test checkpoint ==="
mkdir -p "$CKPT_DIR"
aws s3 sync "s3://$S3_BUCKET/results/bell_test/" "$CKPT_DIR/" \
    --exclude "*.log" --exclude "heartbeat.json" --exclude "aws_completion.json" \
    2>/dev/null || echo "(no prior checkpoint)"
ls -la "$CKPT_DIR/" 2>/dev/null
upload_log

echo "=== Step 6: Starting background sync ==="
(while true; do
    sleep 120
    aws s3 sync "$CKPT_DIR/" "s3://$S3_BUCKET/results/bell_test/" 2>/dev/null || true
    aws s3 sync "$WORKDIR/nbody/bell_test_results/" "s3://$S3_BUCKET/results/bell_test/plots/" 2>/dev/null || true
    upload_log
done) &
LOG_SYNC_PID=$!

echo "=== Step 7: Running bell_test.py ==="
echo "Command: $PYTHON -u nbody/bell_test.py --n-samples 50000 --n-chsh-samples 200000 --n-angles 72 --n-bootstrap 1000 --checkpoint-dir $CKPT_DIR"
echo "=== $(date -u) ==="

cd "$WORKDIR"
$PYTHON -u nbody/bell_test.py \
    --n-samples 50000 \
    --n-chsh-samples 200000 \
    --n-angles 72 \
    --n-bootstrap 1000 \
    --checkpoint-dir "$CKPT_DIR" &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
aws s3 sync "$CKPT_DIR/" "s3://$S3_BUCKET/results/bell_test/" || true
aws s3 sync "$WORKDIR/nbody/bell_test_results/" "s3://$S3_BUCKET/results/bell_test/plots/" || true

upload_log
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/bell_test/full.log" || true

cat > "/tmp/bell_test_completion.json" <<CEOF
{
    "status": "complete",
    "job": "bell_test",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/bell_test_completion.json" \
    "s3://$S3_BUCKET/results/bell_test/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

shutdown -h now 2>/dev/null || true
