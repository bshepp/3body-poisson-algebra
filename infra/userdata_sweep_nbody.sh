#!/bin/bash
# Systematic N-body rank sweep
# N=3-10 through Level 3, N=11-15 through Level 2
# Instance: r6i.8xlarge (32 vCPU, 256 GB RAM)
LOG=/var/log/3body-sweep.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
RESULTS_DIR="results/symbolic_rank"
JOB_TAG="sweep_nbody_ranks"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== N-body Rank Sweep ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

LIVE_LOG_S3="s3://$S3_BUCKET/$RESULTS_DIR/$JOB_TAG/live.log"

upload_log() {
    aws s3 cp "$LOG" "$LIVE_LOG_S3" 2>/dev/null || true
    echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"$JOB_TAG\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
        | aws s3 cp - "s3://$S3_BUCKET/$RESULTS_DIR/$JOB_TAG/heartbeat.json" 2>/dev/null || true
}

sync_all() {
    aws s3 sync "$WORKDIR/results/" "s3://$S3_BUCKET/results/" \
        --include "*.json" --include "*.npy" 2>/dev/null || true
    aws s3 sync "$WORKDIR/nbody/" "s3://$S3_BUCKET/$RESULTS_DIR/$JOB_TAG/checkpoints/" \
        --include "checkpoints_*/rank_results.pkl" \
        --include "checkpoints_*/generators_level*.pkl" \
        --size-only 2>/dev/null || true
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
    sync_all
    upload_log
    exit 0
}
trap shutdown_handler SIGTERM

echo "=== Step 1: Installing dependencies ==="
upload_log

dnf install -y python3 python3-pip python3-devel gcc gmp-devel mpfr-devel libmpc-devel 2>/dev/null \
    || yum install -y python3 python3-pip python3-devel gcc gmp-devel mpfr-devel mpc-devel 2>/dev/null
PYTHON=$(command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip setuptools wheel
$PYTHON -m pip install "sympy>=1.13" numpy scipy gmpy2

echo "=== Step 2: Checking versions ==="
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import gmpy2; print('gmpy2:', gmpy2.version())"
$PYTHON -c "import numpy; print('NumPy:', numpy.__version__)"
echo "CPUs detected: $(nproc)"
upload_log

echo "=== Step 3: Pulling code from S3 ==="
mkdir -p "$WORKDIR/nbody/results/symbolic_rank"
aws s3 sync "s3://$S3_BUCKET/code/nbody/" "$WORKDIR/nbody/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "checkpoints_*/*"

echo "Key files:"
ls -la "$WORKDIR/nbody/symbolic_rank_nbody.py" || echo "symbolic_rank_nbody.py NOT FOUND"
ls -la "$WORKDIR/nbody/exact_growth_nbody.py" || echo "exact_growth_nbody.py NOT FOUND"
ls -la "$WORKDIR/nbody/sweep_nbody_ranks.py" || echo "sweep_nbody_ranks.py NOT FOUND"
upload_log

echo "=== Step 4: Restore any existing results from S3 ==="
mkdir -p "$WORKDIR/results/symbolic_rank"
aws s3 sync "s3://$S3_BUCKET/results/symbolic_rank/" "$WORKDIR/results/symbolic_rank/" \
    --include "rank_N*_d1_1r.json" 2>/dev/null || true
echo "Existing results:"
ls -la "$WORKDIR/results/symbolic_rank/rank_N"*"_d1_1r.json" 2>/dev/null || echo "(none)"

aws s3 sync "s3://$S3_BUCKET/$RESULTS_DIR/$JOB_TAG/checkpoints/" "$WORKDIR/nbody/" \
    --include "checkpoints_*/rank_results.pkl" \
    --include "checkpoints_*/generators_level*.pkl" 2>/dev/null || true
echo "Restored checkpoints:"
ls -lahR "$WORKDIR/nbody/checkpoints_"* 2>/dev/null || echo "(none)"
upload_log

echo "=== Step 5: Background sync (every 5 min) ==="
(while true; do
    sleep 300
    upload_log
    sync_all
done) &
SYNC_PID=$!

NCPU=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || nproc 2>/dev/null || echo 32)
WORKERS=$((NCPU - 1))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

echo "=== Step 6: Running sweep ==="
echo "CPUs: $NCPU, Workers: $WORKERS"
echo "=== $(date -u) ==="
upload_log

cd "$WORKDIR/nbody"
$PYTHON -u sweep_nbody_ranks.py \
    --workers "$WORKERS" \
    --checkpoint-base "$WORKDIR/nbody" \
    --summary "$WORKDIR/results/symbolic_rank/sweep_summary.json" &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Sweep exit code: $EXIT_CODE ==="

echo "=== Final sync ==="
sync_all
aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/$JOB_TAG/full.log" || true

cat > "/tmp/${JOB_TAG}_completion.json" <<CEOF
{
    "status": "complete",
    "job": "$JOB_TAG",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/${JOB_TAG}_completion.json" \
    "s3://$S3_BUCKET/$RESULTS_DIR/$JOB_TAG/aws_completion.json" || true

echo "=== Final local results ==="
ls -la "$WORKDIR/results/symbolic_rank/rank_N"*".json" 2>/dev/null || echo "(none)"
echo "=== Final S3 contents ==="
aws s3 ls "s3://$S3_BUCKET/results/symbolic_rank/" | grep "rank_N" 2>&1
echo "---"
aws s3 ls "s3://$S3_BUCKET/$RESULTS_DIR/$JOB_TAG/" --recursive 2>&1

kill $SYNC_PID 2>/dev/null
echo "=== ALL DONE: $(date -u) ==="

upload_log
shutdown -h now 2>/dev/null || true
