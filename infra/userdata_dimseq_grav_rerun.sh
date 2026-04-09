#!/bin/bash
LOG=/var/log/3body-dimseq-grav-rerun.log
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
echo "=== Gravitational Re-run v5 (SymPy 1.13.3) ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/dimseq_grav_rerun/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"dimseq_grav_rerun\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/results/dimseq_grav_rerun/heartbeat.json" 2>/dev/null || true
}

SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation signal received at $(date -u)"
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        kill -TERM "$SCAN_PID"
        for i in $(seq 1 18); do
            if ! kill -0 "$SCAN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    cd "$WORKDIR/nbody" 2>/dev/null || true
    for d in checkpoints_*; do
        [ -d "$d" ] && aws s3 sync "$d" "s3://$S3_BUCKET/nbody_checkpoints/$d" || true
    done
    aws s3 cp "$WORKDIR/nbody/expansion_dimseq_completion.json" \
        "s3://$S3_BUCKET/results/expansion_dimseq/expansion_dimseq_completion.json" || true
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

echo "Code files:"
ls -la "$WORKDIR/nbody/" 2>/dev/null || echo "nbody/ dir NOT FOUND"

echo "=== Step 4: Purging stale checkpoints ==="
rm -rf "$WORKDIR/nbody/checkpoints_N3_d2_1r" 2>/dev/null || true
rm -rf "$WORKDIR/checkpoints"* 2>/dev/null || true
rm -rf "$WORKDIR/3d/checkpoints"* 2>/dev/null || true
echo "Remaining checkpoint dirs in nbody/:"
ls -d "$WORKDIR/nbody/checkpoints_"* 2>/dev/null || echo "(none)"

echo "=== Step 5: Pulling manifest ==="
cd "$WORKDIR/nbody"
aws s3 cp "s3://$S3_BUCKET/results/expansion_dimseq/expansion_dimseq_completion.json" \
    "$WORKDIR/nbody/expansion_dimseq_completion.json" 2>/dev/null || echo "No manifest found"

upload_log

echo "=== Step 6: Starting background log sync ==="
(while true; do sleep 60; upload_log; done) &
LOG_SYNC_PID=$!

echo "=== Step 7: Running gravitational scenarios ==="
echo "=== $(date -u) ==="

cd "$WORKDIR/nbody"
$PYTHON -u run_expansion_dimseq.py --category gravitational --max-level 3 --samples 1000 &
SCAN_PID=$!
echo "Orchestrator PID: $SCAN_PID"
upload_log

wait $SCAN_PID
EXIT_CODE=$?
SCAN_PID=""
echo "=== Orchestrator exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
cd "$WORKDIR/nbody"
for d in checkpoints_*; do
    [ -d "$d" ] && aws s3 sync "$d" "s3://$S3_BUCKET/nbody_checkpoints/$d" || true
done

aws s3 cp expansion_dimseq_completion.json \
    "s3://$S3_BUCKET/results/expansion_dimseq/expansion_dimseq_completion.json" || true

upload_log
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/dimseq_grav_rerun/full.log" || true

cat > "$WORKDIR/nbody/aws_grav_rerun_completion.json" <<CEOF
{
    "status": "complete",
    "job": "dimseq_grav_rerun_v5",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "$WORKDIR/nbody/aws_grav_rerun_completion.json" \
    "s3://$S3_BUCKET/results/dimseq_grav_rerun/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="
