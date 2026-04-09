#!/bin/bash
LOG=/var/log/3body-symbolic-rank.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
RESULTS_DIR="results/symbolic_rank"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== Symbolic Rank Over Q ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/live.log" 2>/dev/null || true
}

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
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR"
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "checkpoints*"

echo "Key files:"
ls -la "$WORKDIR/symbolic_rank.py" || echo "symbolic_rank.py NOT FOUND"
ls -la "$WORKDIR/exact_growth.py" || echo "exact_growth.py NOT FOUND"
upload_log

echo "=== Step 4: Background log sync ==="
(while true; do
    sleep 300
    upload_log
done) &
SYNC_PID=$!

echo "=== Step 5: Running symbolic rank computation ==="
upload_log

cd "$WORKDIR"

$PYTHON symbolic_rank.py --symbolic \
    --output "$WORKDIR/$RESULTS_DIR/rank_symbolic.json" \
    2>&1 | tee -a "$LOG"

EXIT_CODE=$?
echo "=== Computation finished with exit code $EXIT_CODE ==="

echo "=== Step 6: Uploading results ==="
aws s3 sync "$WORKDIR/$RESULTS_DIR/" "s3://$S3_BUCKET/$RESULTS_DIR/" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/full.log" || true

echo "{\"instance\": \"$INSTANCE_ID\", \"exit_code\": $EXIT_CODE, \"completed\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    | aws s3 cp - "s3://$S3_BUCKET/$RESULTS_DIR/aws_completion.json" || true

kill $SYNC_PID 2>/dev/null || true

echo "=== Shutting down ==="
shutdown -h now
