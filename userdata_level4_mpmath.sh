#!/bin/bash
LOG=/var/log/3body-level4-mpmath.log
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
echo "=== Level 4 Exact Rank (mpmath) ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/level4_mpmath/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"level4_mpmath\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/results/level4_mpmath/heartbeat.json" 2>/dev/null || true
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
    aws s3 sync "$WORKDIR/results/level4_mpmath/" "s3://$S3_BUCKET/results/level4_mpmath/" 2>/dev/null || true
    aws s3 cp "$WORKDIR/checkpoints/level4_derivs.pkl" \
        "s3://$S3_BUCKET/checkpoints/level4_derivs.pkl" 2>/dev/null || true
    upload_log
    exit 0
}
trap shutdown_handler SIGTERM

echo "=== Step 1: Installing dependencies ==="
upload_log

yum install -y python3 python3-pip python3-devel gcc gcc-c++ gmp-devel mpfr-devel mpc-devel
amazon-linux-extras install python3.8 -y 2>/dev/null || true
PYTHON=$(command -v python3.8 || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip 2>/dev/null || true
$PYTHON -m pip install sympy==1.13.3 numpy mpmath gmpy2

echo "=== Step 2: Checking versions ==="
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import mpmath; print('mpmath:', mpmath.__version__)"
$PYTHON -c "import gmpy2; print('gmpy2:', gmpy2.version)" || echo "gmpy2 NOT available (will use pure mpmath)"
$PYTHON -c "import numpy; print('NumPy:', numpy.__version__)"
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR"
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "checkpoints*" --exclude "nbody/*"

echo "Key files:"
ls -la "$WORKDIR/level4_mpmath_rank.py" 2>/dev/null || echo "level4_mpmath_rank.py NOT FOUND"
ls -la "$WORKDIR/exact_growth.py" 2>/dev/null || echo "exact_growth.py NOT FOUND"
upload_log

echo "=== Step 4: Pulling Level 3 checkpoint ==="
mkdir -p "$WORKDIR/checkpoints"
aws s3 cp "s3://$S3_BUCKET/checkpoints/level_3.pkl" "$WORKDIR/checkpoints/level_3.pkl"
ls -la "$WORKDIR/checkpoints/level_3.pkl" || { echo "FATAL: level_3.pkl not found"; upload_log; exit 1; }

aws s3 cp "s3://$S3_BUCKET/results/level4_mpmath/rank_checkpoint.pkl" \
    "$WORKDIR/results/level4_mpmath/rank_checkpoint.pkl" 2>/dev/null || echo "(no prior rank checkpoint)"

echo "=== Step 4b: Pulling cached derivatives (avoids 3h rebuild) ==="
aws s3 cp "s3://$S3_BUCKET/checkpoints/level4_derivs.pkl" "$WORKDIR/checkpoints/level4_derivs.pkl" \
    2>/dev/null || echo "(no cached derivatives — will compute from scratch)"
ls -la "$WORKDIR/checkpoints/level4_derivs.pkl" 2>/dev/null
upload_log

echo "=== Step 5: Starting background sync ==="
(while true; do
    sleep 120
    aws s3 sync "$WORKDIR/results/level4_mpmath/" "s3://$S3_BUCKET/results/level4_mpmath/" 2>/dev/null || true
    aws s3 cp "$WORKDIR/checkpoints/level4_derivs.pkl" \
        "s3://$S3_BUCKET/checkpoints/level4_derivs.pkl" 2>/dev/null || true
    upload_log
done) &
LOG_SYNC_PID=$!

RESUME_FLAG=""
if [ -f "$WORKDIR/results/level4_mpmath/rank_checkpoint.pkl" ]; then
    RESUME_FLAG="--resume"
    echo "Found prior checkpoint, will resume"
fi

echo "=== Step 6: Running level4_mpmath_rank.py ==="
echo "Command: $PYTHON -u level4_mpmath_rank.py --dps 50 --max-rows 15000 --plateau 200 $RESUME_FLAG"
echo "=== $(date -u) ==="

cd "$WORKDIR"
$PYTHON -u level4_mpmath_rank.py --dps 50 --max-rows 15000 --plateau 200 $RESUME_FLAG &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
aws s3 sync "$WORKDIR/results/level4_mpmath/" "s3://$S3_BUCKET/results/level4_mpmath/" || true

upload_log
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/level4_mpmath/full.log" || true

cat > "/tmp/level4_completion.json" <<CEOF
{
    "status": "complete",
    "job": "level4_mpmath",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/level4_completion.json" \
    "s3://$S3_BUCKET/results/level4_mpmath/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

shutdown -h now 2>/dev/null || true
