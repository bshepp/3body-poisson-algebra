#!/bin/bash
# ==========================================================================
# AWS Spot Instance -- Multi-System Universality Survey (Atlas Scans)
#
# Runs targeted_adaptive_scan.py for all atlas-enabled scenarios from
# expansion_configs.py across the 6 standard targeted regions.
#
# Instance: c6i.8xlarge (32 vCPUs, 64 GB RAM)
# ==========================================================================

LOG=/var/log/3body-expansion-atlas.log
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
echo "=== Multi-System Universality Survey -- Atlas Scans ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "CPUs: $(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())' 2>/dev/null || echo unknown)"
echo "RAM:  $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo unknown)"

# ---------- Trap SIGTERM for graceful shutdown ----------------------------
SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation signal received at $(date -u)"
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        echo "[SIGTERM] Forwarding to orchestrator PID $SCAN_PID"
        kill -TERM "$SCAN_PID"
        for i in $(seq 1 18); do
            if ! kill -0 "$SCAN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    echo "[SIGTERM] Emergency sync..."
    cd "$WORKDIR"
    aws s3 sync atlas_targeted/ "s3://$S3_BUCKET/atlas_targeted/" || true
    aws s3 cp expansion_atlas_completion.json \
        "s3://$S3_BUCKET/results/expansion_atlas/expansion_atlas_completion.json" || true
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/expansion_atlas/full.log" || true
    echo "[SIGTERM] Shutdown complete at $(date -u)"
    exit 0
}
trap shutdown_handler SIGTERM

# ---------- Install dependencies ------------------------------------------
dnf install -y python3 python3-pip 2>/dev/null || yum install -y python3 python3-pip
pip3 install sympy numpy scipy matplotlib

echo "Python: $(python3 --version)"
python3 -c "import sympy; print('SymPy:', sympy.__version__)"

# ---------- Pull code from S3 --------------------------------------------
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Pulling code ==="
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "checkpoints*"

echo "Code files:"
ls -la "$WORKDIR/"

# ---------- Pull existing scan data (for resume) -------------------------
echo "=== Pulling existing scan data for resume ==="
aws s3 sync "s3://$S3_BUCKET/atlas_targeted/" "$WORKDIR/atlas_targeted/" || true

aws s3 cp "s3://$S3_BUCKET/results/expansion_atlas/expansion_atlas_completion.json" \
    "$WORKDIR/expansion_atlas_completion.json" 2>/dev/null || true

for cp_file in $(find "$WORKDIR/atlas_targeted/" -name "checkpoint.json" 2>/dev/null); do
    echo "  Found checkpoint: $cp_file"
done

# ---------- Background log sync (every 60s) -------------------------------
sync_logs() {
    while true; do
        sleep 60
        aws s3 cp "$LOG" \
            "s3://$S3_BUCKET/results/expansion_atlas/live.log" 2>/dev/null || true
        echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"job\": \"expansion_atlas\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
            | aws s3 cp - "s3://$S3_BUCKET/results/expansion_atlas/heartbeat.json" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# ---------- Run the orchestrator ------------------------------------------
echo ""
echo "=================================================================="
echo "=== Starting expansion atlas orchestrator ==="
echo "=== $(date -u) ==="
echo "=================================================================="

cd "$WORKDIR"
python3 -u run_expansion_atlas.py &
SCAN_PID=$!
wait $SCAN_PID
EXIT_CODE=$?
SCAN_PID=""

echo "=== Orchestrator exit code: $EXIT_CODE ==="

# ---------- Final sync (everything, with retries) -------------------------
echo "=== Final data sync (with retries) ==="
for attempt in 1 2 3; do
    echo "  Sync attempt $attempt..."
    aws s3 sync "$WORKDIR/atlas_targeted/" "s3://$S3_BUCKET/atlas_targeted/" && break
    sleep 10
done

aws s3 cp "$WORKDIR/expansion_atlas_completion.json" \
    "s3://$S3_BUCKET/results/expansion_atlas/expansion_atlas_completion.json" || true
aws s3 cp "$WORKDIR/expansion_atlas_summary.json" \
    "s3://$S3_BUCKET/results/expansion_atlas/expansion_atlas_summary.json" 2>/dev/null || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/expansion_atlas/full.log" || true

# ---------- Write completion marker ---------------------------------------
cat > "$WORKDIR/aws_expansion_atlas_completion.json" <<CEOF
{
    "status": "complete",
    "job": "expansion_atlas",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "$WORKDIR/aws_expansion_atlas_completion.json" \
    "s3://$S3_BUCKET/results/expansion_atlas/aws_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null

echo "=== ALL DONE: $(date -u) ==="
echo "=== Exit code: $EXIT_CODE ==="
