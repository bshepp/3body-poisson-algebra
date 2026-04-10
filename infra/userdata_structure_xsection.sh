#!/bin/bash
# 1D cross-section: numerical structure constants along mu sweep
# Runs 1/r and r^4 potentials sequentially
LOG=/var/log/3body-structure-xsection.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
RESULTS_DIR="results/structure_cross_section"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== Structure Cross-Section ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"structure_xsection\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/$RESULTS_DIR/heartbeat.json" 2>/dev/null || true
}

MAIN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Received at $(date -u)"
    if [ -n "$MAIN_PID" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
        kill -TERM "$MAIN_PID"
        for i in $(seq 1 30); do
            if ! kill -0 "$MAIN_PID" 2>/dev/null; then break; fi
            sleep 2
        done
    fi
    aws s3 sync "$WORKDIR/nbody/results/structure_cross_section/" \
        "s3://$S3_BUCKET/$RESULTS_DIR/" 2>/dev/null || true
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
$PYTHON -m pip install sympy==1.13.3 numpy scipy gmpy2

echo "=== Step 2: Checking versions ==="
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import gmpy2; print('gmpy2:', gmpy2.version())"
$PYTHON -c "import numpy; print('NumPy:', numpy.__version__)"
$PYTHON -c "import scipy; print('SciPy:', scipy.__version__)"
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR/nbody"
aws s3 sync "s3://$S3_BUCKET/code/nbody/" "$WORKDIR/nbody/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "checkpoints_*/*"

echo "Key files:"
ls -la "$WORKDIR/nbody/structure_cross_section.py" || echo "structure_cross_section.py NOT FOUND"
ls -la "$WORKDIR/nbody/symbolic_rank_nbody.py" || echo "symbolic_rank_nbody.py NOT FOUND"
ls -la "$WORKDIR/nbody/exact_growth_nbody.py" || echo "exact_growth_nbody.py NOT FOUND"
upload_log

echo "=== Step 4: Create results directory ==="
mkdir -p "$WORKDIR/nbody/results/structure_cross_section"

echo "=== Step 5: Background sync (every 5 min) ==="
(while true; do
    sleep 300
    upload_log
    aws s3 sync "$WORKDIR/nbody/results/structure_cross_section/" \
        "s3://$S3_BUCKET/$RESULTS_DIR/" 2>/dev/null || true
done) &
SYNC_PID=$!

cd "$WORKDIR/nbody"

echo "=== Step 6a: Running 1/r cross-section ==="
echo "Command: $PYTHON -u structure_cross_section.py --potential 1/r --n-mu 200 --max-level 2"
echo "=== $(date -u) ==="
upload_log

$PYTHON -u structure_cross_section.py \
    --potential "1/r" --n-mu 200 --max-level 2 --n-samples 300 &
MAIN_PID=$!
wait $MAIN_PID
EXIT_1R=$?
MAIN_PID=""
echo "=== 1/r exit code: $EXIT_1R ==="

aws s3 sync "$WORKDIR/nbody/results/structure_cross_section/" \
    "s3://$S3_BUCKET/$RESULTS_DIR/" || true
upload_log

echo "=== Step 6b: Running r^4 cross-section ==="
echo "Command: $PYTHON -u structure_cross_section.py --potential r^4 --n-mu 200 --max-level 2"
echo "=== $(date -u) ==="
upload_log

$PYTHON -u structure_cross_section.py \
    --potential "r^4" --n-mu 200 --max-level 2 --n-samples 300 &
MAIN_PID=$!
wait $MAIN_PID
EXIT_R4=$?
MAIN_PID=""
echo "=== r^4 exit code: $EXIT_R4 ==="

echo "=== Step 6c: Running 1/r^4 cross-section ==="
echo "Command: $PYTHON -u structure_cross_section.py --potential composite --composite -1 4 --n-mu 200 --max-level 2"
echo "=== $(date -u) ==="
upload_log

$PYTHON -u structure_cross_section.py \
    --potential "composite" --composite "-1" "4" --n-mu 200 --max-level 2 --n-samples 300 &
MAIN_PID=$!
wait $MAIN_PID
EXIT_1R4=$?
MAIN_PID=""
echo "=== 1/r^4 exit code: $EXIT_1R4 ==="

echo "=== Final sync ==="
aws s3 sync "$WORKDIR/nbody/results/structure_cross_section/" \
    "s3://$S3_BUCKET/$RESULTS_DIR/" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/full.log" || true

cat > "/tmp/xsection_completion.json" <<CEOF
{
    "status": "complete",
    "job": "structure_cross_section",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_codes": {"1r": $EXIT_1R, "r4": $EXIT_R4, "1r4": $EXIT_1R4},
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/xsection_completion.json" \
    "s3://$S3_BUCKET/$RESULTS_DIR/aws_completion.json" || true

kill $SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

shutdown -h now 2>/dev/null || true
