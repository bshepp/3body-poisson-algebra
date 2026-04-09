#!/bin/bash
# ==========================================================================
# Robust AWS Spot Instance Launcher for Targeted Adaptive Scans
#
# Handles:
#   - SIGTERM from spot reclamation (2-min warning)
#   - Resume from S3 checkpoint on fresh instance
#   - Periodic log + data sync to S3
#   - Final verification and data collection
# ==========================================================================

LOG=/var/log/3body-targeted.log
exec > "$LOG" 2>&1
set -x

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "=== 3-Body Targeted Adaptive Scans ==="
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
echo "nproc reports: $(nproc)"
echo "/proc/cpuinfo reports: $(grep -c ^processor /proc/cpuinfo)"

# ---------- Trap SIGTERM for graceful shutdown ----------------------------
SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation signal received at $(date -u)"
    # Forward SIGTERM to the running scan so it saves + syncs
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        echo "[SIGTERM] Forwarding to scan PID $SCAN_PID"
        kill -TERM "$SCAN_PID"
        # Give it up to 90s to save and sync (spot gives ~120s total)
        for i in $(seq 1 18); do
            if ! kill -0 "$SCAN_PID" 2>/dev/null; then break; fi
            sleep 5
        done
    fi
    # Final emergency sync of everything
    echo "[SIGTERM] Emergency sync..."
    aws s3 sync "$WORKDIR/atlas_targeted/" "s3://$S3_BUCKET/atlas_targeted/" || true
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/targeted_$(date +%s).log" || true
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

# Show any existing checkpoints
for cp in $(find "$WORKDIR/atlas_targeted/" -name "checkpoint.json" 2>/dev/null); do
    echo "  Found checkpoint: $cp"
    cat "$cp"
done

# ---------- Background log sync (every 60s) -------------------------------
sync_logs() {
    while true; do
        sleep 60
        aws s3 cp "$LOG" "s3://$S3_BUCKET/results/targeted_live.log" 2>/dev/null || true
        # Sync a global status summary
        echo "{\"instance\": \"$INSTANCE_ID\", \"type\": \"$INSTANCE_TYPE\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"uptime\": \"$(uptime)\"}" \
            | aws s3 cp - "s3://$S3_BUCKET/results/targeted_heartbeat.json" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# ---------- Run reference (uncharged) 1/r2 scans -------------------------
echo ""
echo "=================================================================="
echo "=== Starting targeted scans: reference 1/r2 ==="
echo "=== Started: $(date -u) ==="
echo "=================================================================="

python3 -u targeted_adaptive_scan.py \
    --potential 1/r2 \
    2>&1 | tee "$WORKDIR/targeted_ref.log" &
SCAN_PID=$!
wait $SCAN_PID
REF_EXIT=$?
SCAN_PID=""

echo "=== Reference scan exit code: $REF_EXIT ==="

# Sync everything after reference scan
aws s3 sync "$WORKDIR/atlas_targeted/" "s3://$S3_BUCKET/atlas_targeted/" || true
aws s3 cp "$WORKDIR/targeted_ref.log" "s3://$S3_BUCKET/results/targeted_ref.log" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/targeted_full.log" || true

# ---------- Run charged (helium) 1/r2 scans -------------------------------
echo ""
echo "=================================================================="
echo "=== Starting targeted scans: helium 1/r2 ==="
echo "=== Started: $(date -u) ==="
echo "=================================================================="

python3 -u targeted_adaptive_scan.py \
    --potential 1/r2 \
    --charges 2 -1 -1 \
    2>&1 | tee "$WORKDIR/targeted_chg.log" &
SCAN_PID=$!
wait $SCAN_PID
CHG_EXIT=$?
SCAN_PID=""

echo "=== Helium scan exit code: $CHG_EXIT ==="

# Sync after charged scan
aws s3 sync "$WORKDIR/atlas_targeted/" "s3://$S3_BUCKET/atlas_targeted/" || true
aws s3 cp "$WORKDIR/targeted_chg.log" "s3://$S3_BUCKET/results/targeted_chg.log" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/targeted_full.log" || true

# ---------- Run analysis --------------------------------------------------
echo "=== Running analysis ==="
python3 -u targeted_adaptive_scan.py --analyze --potential 1/r2 2>&1
python3 -u targeted_adaptive_scan.py --analyze --potential 1/r2 --charges 2 -1 -1 2>&1

# ---------- Final sync (everything, with retries) -------------------------
echo "=== Final data sync (with retries) ==="
for attempt in 1 2 3; do
    echo "  Sync attempt $attempt..."
    aws s3 sync "$WORKDIR/atlas_targeted/" "s3://$S3_BUCKET/atlas_targeted/" && break
    sleep 10
done

# Upload final logs
aws s3 cp "$WORKDIR/targeted_ref.log" "s3://$S3_BUCKET/results/targeted_ref.log" || true
aws s3 cp "$WORKDIR/targeted_chg.log" "s3://$S3_BUCKET/results/targeted_chg.log" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/targeted_full.log" || true

# ---------- Write completion marker ---------------------------------------
cat > "$WORKDIR/completion.json" <<CEOF
{
    "status": "complete",
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "ref_exit_code": $REF_EXIT,
    "chg_exit_code": $CHG_EXIT,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "$WORKDIR/completion.json" "s3://$S3_BUCKET/results/targeted_completion.json" || true

# Kill background log sync
kill $LOG_SYNC_PID 2>/dev/null

echo "=== ALL DONE: $(date -u) ==="
echo "=== Exit codes: ref=$REF_EXIT chg=$CHG_EXIT ==="
