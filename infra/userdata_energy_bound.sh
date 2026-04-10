#!/bin/bash
# Energy Bound Search: commutant computation, Casimir, sign analysis
# Instance: r6i.xlarge (4 vCPU, 32 GB RAM) — modest memory needs
LOG=/var/log/3body-energy-bound.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
RESULTS_DIR="results/energy_bound"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== Energy Bound Search ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/live.log" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"energy_bound\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/$RESULTS_DIR/heartbeat.json" 2>/dev/null || true
}

MAIN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Received at $(date -u)"
    if [ -n "$MAIN_PID" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
        kill -TERM "$MAIN_PID"
        for i in $(seq 1 120); do
            if ! kill -0 "$MAIN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    aws s3 sync "$WORKDIR/nbody/checkpoints_energy_bound/" \
        "s3://$S3_BUCKET/$RESULTS_DIR/checkpoints/" --size-only 2>/dev/null || true
    aws s3 sync "$WORKDIR/results/energy_bound/" \
        "s3://$S3_BUCKET/$RESULTS_DIR/" --include "*.json" 2>/dev/null || true
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
ls -la "$WORKDIR/nbody/energy_bound_search.py" || echo "energy_bound_search.py NOT FOUND"
ls -la "$WORKDIR/nbody/exact_growth_nbody.py" || echo "exact_growth_nbody.py NOT FOUND"
ls -la "$WORKDIR/nbody/quantum_algebra.py" || echo "quantum_algebra.py NOT FOUND"
ls -la "$WORKDIR/nbody/identify_117th.py" || echo "identify_117th.py NOT FOUND"
upload_log

echo "=== Step 4: Restore checkpoints from S3 ==="
mkdir -p "$WORKDIR/nbody/checkpoints_energy_bound"
mkdir -p "$WORKDIR/results/energy_bound"

aws s3 sync "s3://$S3_BUCKET/$RESULTS_DIR/checkpoints/" \
    "$WORKDIR/nbody/checkpoints_energy_bound/" 2>/dev/null || true
echo "Checkpoints restored:"
ls -lah "$WORKDIR/nbody/checkpoints_energy_bound/" 2>/dev/null || echo "(none)"
upload_log

echo "=== Step 5: Background log sync (every 3 min) ==="
(while true; do
    sleep 180
    upload_log
    aws s3 sync "$WORKDIR/nbody/checkpoints_energy_bound/" \
        "s3://$S3_BUCKET/$RESULTS_DIR/checkpoints/" --size-only 2>/dev/null || true
done) &
SYNC_PID=$!

echo "=== Step 6: Running energy bound search ==="
echo "Command: $PYTHON -u energy_bound_search.py"
echo "=== $(date -u) ==="
upload_log

cd "$WORKDIR/nbody"
$PYTHON -u energy_bound_search.py &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
aws s3 sync "$WORKDIR/nbody/checkpoints_energy_bound/" \
    "s3://$S3_BUCKET/$RESULTS_DIR/checkpoints/" --size-only || true
aws s3 sync "$WORKDIR/results/energy_bound/" \
    "s3://$S3_BUCKET/$RESULTS_DIR/" --include "*.json" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/full.log" || true

cat > "/tmp/energy_bound_completion.json" <<CEOF
{
    "status": "complete",
    "job": "energy_bound_search",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/energy_bound_completion.json" \
    "s3://$S3_BUCKET/$RESULTS_DIR/aws_completion.json" || true

echo "=== Verifying S3 sync ==="
echo "--- S3 results ---"
aws s3 ls "s3://$S3_BUCKET/$RESULTS_DIR/" --recursive 2>&1
echo "--- Checking final results ---"
aws s3 ls "s3://$S3_BUCKET/$RESULTS_DIR/energy_bound_results.json" 2>&1

kill $SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

upload_log
shutdown -h now 2>/dev/null || true
