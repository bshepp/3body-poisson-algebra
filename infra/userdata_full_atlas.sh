#!/bin/bash
LOG=/var/log/3body-full-atlas.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# ---- CONFIGURABLE: set these before launching ----
ATLAS_ARGS="__ATLAS_ARGS__"
JOB_NAME="__JOB_NAME__"
# ---------------------------------------------------

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== Full Atlas Scan: $JOB_NAME ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "Args:     $ATLAS_ARGS"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/atlas_full/${JOB_NAME}/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"$JOB_NAME\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/results/atlas_full/${JOB_NAME}/heartbeat.json" 2>/dev/null || true
}

SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation signal received at $(date -u)"
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        kill -TERM "$SCAN_PID"
        for i in $(seq 1 30); do
            if ! kill -0 "$SCAN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    aws s3 sync "$WORKDIR/atlas_full/" "s3://$S3_BUCKET/atlas_full/" 2>/dev/null || true
    upload_log
    exit 0
}
trap shutdown_handler SIGTERM

echo "=== Step 1: Installing dependencies ==="
upload_log

yum install -y python3 python3-pip python3-devel gcc
amazon-linux-extras install python3.8 -y 2>/dev/null || true
PYTHON=$(command -v python3.8 || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip 2>/dev/null || true
$PYTHON -m pip install sympy==1.13.3 numpy scipy

echo "=== Step 2: Checking versions ==="
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import numpy; print('NumPy:', numpy.__version__)"
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR"
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "checkpoints*"

echo "Key files:"
ls -la "$WORKDIR/full_atlas_scan.py" 2>/dev/null || echo "full_atlas_scan.py NOT FOUND"
ls -la "$WORKDIR/stability_atlas.py" 2>/dev/null || echo "stability_atlas.py NOT FOUND"
ls -la "$WORKDIR/exact_growth.py" 2>/dev/null || echo "exact_growth.py NOT FOUND"
ls -la "$WORKDIR/nbody/expansion_configs.py" 2>/dev/null || echo "expansion_configs.py NOT FOUND"
upload_log

echo "=== Step 4: Resume any prior checkpoint ==="
mkdir -p "$WORKDIR/atlas_full"
aws s3 sync "s3://$S3_BUCKET/atlas_full/" "$WORKDIR/atlas_full/" \
    --exclude "*.log" 2>/dev/null || true
echo "Existing atlas_full contents:"
find "$WORKDIR/atlas_full" -name "checkpoint.json" -exec echo {} \; -exec cat {} \; 2>/dev/null || echo "(none)"
upload_log

echo "=== Step 5: Starting background sync ==="
(while true; do
    sleep 120
    aws s3 sync "$WORKDIR/atlas_full/" "s3://$S3_BUCKET/atlas_full/" 2>/dev/null || true
    upload_log
done) &
LOG_SYNC_PID=$!

echo "=== Step 6: Running atlas scan ==="
echo "Command: $PYTHON -u $WORKDIR/full_atlas_scan.py $ATLAS_ARGS --output-dir $WORKDIR/atlas_full"
echo "=== $(date -u) ==="

cd "$WORKDIR"
$PYTHON -u full_atlas_scan.py $ATLAS_ARGS --output-dir "$WORKDIR/atlas_full" &
SCAN_PID=$!
echo "Scanner PID: $SCAN_PID"
upload_log

wait $SCAN_PID
EXIT_CODE=$?
SCAN_PID=""
echo "=== Scanner exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
aws s3 sync "$WORKDIR/atlas_full/" "s3://$S3_BUCKET/atlas_full/" || true

upload_log
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/atlas_full/${JOB_NAME}/full.log" || true

cat > "/tmp/atlas_completion.json" <<CEOF
{
    "status": "complete",
    "job": "$JOB_NAME",
    "atlas_args": "$ATLAS_ARGS",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/atlas_completion.json" \
    "s3://$S3_BUCKET/results/atlas_full/${JOB_NAME}/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

shutdown -h now 2>/dev/null || true
