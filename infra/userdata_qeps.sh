#!/bin/bash
# Q(eps) symbolic nullspace on (4,3) binary stratum
# planar 1/r d=2 level-3 collision syzygy (rank/nullity over Q(eps))
LOG=/var/log/3body-qeps.log
exec > "$LOG" 2>&1
set -x
set -o pipefail
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
RESULTS_DIR="results/collision_syzygy_qeps"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== Q(eps) symbolic nullspace job ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

upload_log() {
    aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/live.log" 2>/dev/null || true
    echo "{\"instance\":\"$INSTANCE_ID\",\"type\":\"$INSTANCE_TYPE\",\"job\":\"qeps_nullspace\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"uptime\":\"$(uptime)\"}" \
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
    aws s3 cp "$WORKDIR/collision_syzygy_qeps.json" \
        "s3://$S3_BUCKET/$RESULTS_DIR/collision_syzygy_qeps.json" 2>/dev/null || true
    upload_log
    exit 0
}
trap shutdown_handler SIGTERM

echo "=== Step 1: dependencies ==="
upload_log
yum install -y python3 python3-pip python3-devel gcc gmp-devel mpfr-devel mpc-devel
# AL2 ships Python 3.7; SymPy >=1.11 needs 3.8+. Pull python3.8 from extras.
amazon-linux-extras install python3.8 -y 2>/dev/null || true
PYTHON=$(command -v python3.8 || command -v python3)
echo "Using $PYTHON ($($PYTHON --version 2>&1))"
$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip 2>/dev/null || true
$PYTHON -m pip install sympy==1.13.3 numpy gmpy2

$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"
$PYTHON -c "import gmpy2; print('gmpy2:', gmpy2.version())"

echo "=== Step 2: sync code + checkpoint from S3 ==="
mkdir -p "$WORKDIR/checkpoints"
aws s3 cp "s3://$S3_BUCKET/code/3body/collision_syzygy_v2.py"      "$WORKDIR/"
aws s3 cp "s3://$S3_BUCKET/code/3body/collision_syzygy_v2_qeps.py" "$WORKDIR/"
aws s3 cp "s3://$S3_BUCKET/code/3body/exact_growth.py"             "$WORKDIR/"
aws s3 cp "s3://$S3_BUCKET/code/3body/checkpoints/level_3.pkl"     "$WORKDIR/checkpoints/"

ls -la "$WORKDIR/" "$WORKDIR/checkpoints/"
upload_log

echo "=== Step 3: background sync (every 5 min) ==="
(while true; do
    sleep 300
    upload_log
    if [ -f "$WORKDIR/collision_syzygy_qeps.json" ]; then
        aws s3 cp "$WORKDIR/collision_syzygy_qeps.json" \
            "s3://$S3_BUCKET/$RESULTS_DIR/collision_syzygy_qeps.json" 2>/dev/null || true
    fi
done) &
SYNC_PID=$!

echo "=== Step 4: run Q(eps) job ==="
cd "$WORKDIR"
$PYTHON -u collision_syzygy_v2_qeps.py &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"
upload_log

wait $MAIN_PID
EXIT_CODE=$?
MAIN_PID=""
echo "=== Exit code: $EXIT_CODE ==="

echo "=== Final upload ==="
if [ -f "$WORKDIR/collision_syzygy_qeps.json" ]; then
    aws s3 cp "$WORKDIR/collision_syzygy_qeps.json" \
        "s3://$S3_BUCKET/$RESULTS_DIR/collision_syzygy_qeps.json" || true
fi
aws s3 cp "$LOG" "s3://$S3_BUCKET/$RESULTS_DIR/full.log" || true

cat > "/tmp/qeps_completion.json" <<CEOF
{
    "status": "complete",
    "job": "qeps_nullspace",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "/tmp/qeps_completion.json" \
    "s3://$S3_BUCKET/$RESULTS_DIR/aws_completion.json" || true

# Auto-shutdown to stop billing once done
shutdown -h +5
