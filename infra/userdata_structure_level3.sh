#!/bin/bash
# Level-3 structure extraction: exact structure constants for rank-116 algebra
# N=3, d=2, 1/r potential with --structure flag
LOG=/var/log/3body-structure-level3.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
RESULTS_DIR="results/algebra_structure/N3_d2_1r_level3"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== Level-3 Structure Extraction ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"structure_level3\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/$RESULTS_DIR/heartbeat.json" 2>/dev/null || true
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
    aws s3 sync "$WORKDIR/nbody/results/" "s3://$S3_BUCKET/results/" \
        --include "*.json" --include "*.npy" 2>/dev/null || true
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

echo "=== Step 4: Create results directory ==="
mkdir -p "$WORKDIR/nbody/results/symbolic_rank"
mkdir -p "$WORKDIR/nbody/results/algebra_structure"

echo "=== Step 5: Background log sync (every 5 min) ==="
(while true; do
    sleep 300
    upload_log
    aws s3 sync "$WORKDIR/nbody/results/" "s3://$S3_BUCKET/results/" \
        --include "*.json" --include "*.npy" 2>/dev/null || true
done) &
SYNC_PID=$!

echo "=== Step 6: Running level-3 structure extraction ==="
echo "Command: $PYTHON -u symbolic_rank_nbody.py -N 3 -d 2 --max-level 3 --structure"
echo "=== $(date -u) ==="
upload_log

cd "$WORKDIR/nbody"
$PYTHON -u symbolic_rank_nbody.py -N 3 -d 2 --max-level 3 --structure &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
aws s3 sync "$WORKDIR/nbody/results/" "s3://$S3_BUCKET/results/" \
    --include "*.json" --include "*.npy" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/full.log" || true

cat > "/tmp/structure_level3_completion.json" <<CEOF
{
    "status": "complete",
    "job": "structure_level3",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/structure_level3_completion.json" \
    "s3://$S3_BUCKET/$RESULTS_DIR/aws_completion.json" || true

kill $SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

shutdown -h now 2>/dev/null || true
