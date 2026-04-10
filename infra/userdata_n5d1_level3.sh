#!/bin/bash
# N=5 d=1 level 3: exact symbolic rank with per-level checkpointing
# Instance: r6i.4xlarge (16 vCPU, 128 GB RAM)
LOG=/var/log/3body-n5d1-level3.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
RESULTS_DIR="results/symbolic_rank"
CHECKPOINT_DIR="$WORKDIR/nbody/checkpoints_n5d1_level3"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== N=5 d=1 Level-3 Symbolic Rank ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

LIVE_LOG_S3="s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/live.log"

upload_log() {
    aws s3 cp "$LOG" "$LIVE_LOG_S3" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"N5_d1_level3\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/heartbeat.json" 2>/dev/null || true
}

sync_checkpoints() {
    aws s3 sync "$CHECKPOINT_DIR/" "s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/checkpoints/" \
        --size-only 2>/dev/null || true
}

sync_results() {
    aws s3 sync "$WORKDIR/nbody/results/" "s3://$S3_BUCKET/results/" \
        --include "*.json" --include "*.npy" 2>/dev/null || true
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
    sync_checkpoints
    sync_results
    upload_log
    exit 0
}
trap shutdown_handler SIGTERM

echo "=== Step 1: Installing dependencies ==="
upload_log

yum install -y python3 python3-pip python3-devel gcc gmp-devel mpfr-devel mpc-devel
amazon-linux-extras install python3.8 -y 2>/dev/null || true
PYTHON=$(command -v python3.8 || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip 2>/dev/null || true
$PYTHON -m pip install sympy==1.14.0 numpy scipy gmpy2

echo "=== Step 2: Checking versions ==="
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import gmpy2; print('gmpy2:', gmpy2.version())"
$PYTHON -c "import numpy; print('NumPy:', numpy.__version__)"
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR/nbody"
aws s3 sync "s3://$S3_BUCKET/code/nbody/" "$WORKDIR/nbody/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "checkpoints_*/*"

echo "Key files:"
ls -la "$WORKDIR/nbody/symbolic_rank_nbody.py" || echo "symbolic_rank_nbody.py NOT FOUND"
ls -la "$WORKDIR/nbody/exact_growth_nbody.py" || echo "exact_growth_nbody.py NOT FOUND"
upload_log

echo "=== Step 4: Restore checkpoints from S3 ==="
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$WORKDIR/nbody/results/symbolic_rank"
mkdir -p "$WORKDIR/nbody/results/algebra_structure"

aws s3 sync "s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/checkpoints/" "$CHECKPOINT_DIR/" 2>/dev/null || true
echo "Checkpoints restored:"
ls -lah "$CHECKPOINT_DIR/" 2>/dev/null || echo "(none)"
upload_log

echo "=== Step 5: Background sync (every 5 min) ==="
(while true; do
    sleep 300
    upload_log
    sync_checkpoints
    sync_results
done) &
SYNC_PID=$!

echo "=== Step 6: Running N=5 d=1 level 3 ==="
echo "Command: $PYTHON -u symbolic_rank_nbody.py -N 5 -d 1 --max-level 3 --checkpoint-dir $CHECKPOINT_DIR"
echo "=== $(date -u) ==="
upload_log

cd "$WORKDIR/nbody"
$PYTHON -u symbolic_rank_nbody.py -N 5 -d 1 --max-level 3 \
    --checkpoint-dir "$CHECKPOINT_DIR" &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
sync_checkpoints
sync_results
aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/full.log" || true

cat > "/tmp/n5d1_level3_completion.json" <<CEOF
{
    "status": "complete",
    "job": "N5_d1_level3",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/n5d1_level3_completion.json" \
    "s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/aws_completion.json" || true

echo "=== Verifying S3 sync ==="
echo "--- S3 results ---"
aws s3 ls "s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/" --recursive 2>&1
echo "--- S3 checkpoints ---"
aws s3 ls "s3://$S3_BUCKET/$RESULTS_DIR/N5_d1_level3/checkpoints/" 2>&1
echo "--- S3 rank JSON ---"
aws s3 ls "s3://$S3_BUCKET/$RESULTS_DIR/" | grep "N5_d1" 2>&1

SYNC_OK=$?
if [ $SYNC_OK -eq 0 ]; then
    echo "=== S3 sync verified ==="
else
    echo "=== WARNING: S3 sync verification had issues ==="
fi

kill $SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

upload_log
shutdown -h now 2>/dev/null || true
