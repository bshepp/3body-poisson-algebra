#!/bin/bash
# ==========================================================================
# AWS Spot Instance Launcher — Shape Sphere Atlas
#
# Direct (theta, phi_sph) sampling on S^2 for the planar 3-body Poisson
# algebra. Each instance handles one [BLOCK_START, BLOCK_END) row band
# (theta-rows) for the given potential, then syncs blocks to S3.
#
# Required env (passed via launch template / spot fleet user data):
#   POTENTIAL    e.g. "1/r" | "1/r2" | "1/r3" | "harmonic"
#   GRID_THETA   e.g. 400
#   GRID_PHI     e.g. 800
#   BLOCK_START  inclusive theta-row start (0..GRID_THETA)
#   BLOCK_END    exclusive theta-row end
#   EPSILON      e.g. 1e-3
#   WORKERS      e.g. 16
#   S3_BUCKET    e.g. nbody-briansheppard-com
# ==========================================================================

LOG=/var/log/3body-sphere.log
exec > "$LOG" 2>&1
set -x

: "${POTENTIAL:=1/r}"
: "${GRID_THETA:=400}"
: "${GRID_PHI:=800}"
: "${BLOCK_START:=0}"
: "${BLOCK_END:=400}"
: "${EPSILON:=1e-3}"
: "${WORKERS:=16}"
: "${S3_BUCKET:=nbody-briansheppard-com}"

WORKDIR="/opt/3body"
export S3_BUCKET PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")

POT_TAG=$(echo "$POTENTIAL" | tr '/' '_')
TAG="sphere_${POT_TAG}_${BLOCK_START}_${BLOCK_END}"

echo "=== 3-Body Shape Sphere Atlas ==="
echo "Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance:   $INSTANCE_ID ($INSTANCE_TYPE)"
echo "Potential:  $POTENTIAL   eps=$EPSILON"
echo "Grid:       theta x phi = $GRID_THETA x $GRID_PHI"
echo "Block:      [$BLOCK_START, $BLOCK_END)   workers=$WORKERS"

# ---------- Trap SIGTERM for graceful shutdown ----------------------------
SCAN_PID=""
shutdown_handler() {
    echo "[SIGTERM] Spot reclamation at $(date -u)"
    if [ -n "$SCAN_PID" ] && kill -0 "$SCAN_PID" 2>/dev/null; then
        kill -TERM "$SCAN_PID"
        for i in $(seq 1 18); do
            kill -0 "$SCAN_PID" 2>/dev/null || break
            sleep 5
        done
    fi
    aws s3 sync "$WORKDIR/shape_sphere_atlas/" "s3://$S3_BUCKET/shape_sphere_atlas/" || true
    aws s3 cp "$LOG" "s3://$S3_BUCKET/results/${TAG}_$(date +%s).log" || true
    exit 0
}
trap shutdown_handler SIGTERM

# ---------- Install dependencies ------------------------------------------
dnf install -y python3 python3-pip 2>/dev/null || yum install -y python3 python3-pip
pip3 install 'sympy>=1.13.3' numpy scipy

python3 -c "import sympy; print('SymPy:', sympy.__version__)"

# ---------- Pull code from S3 --------------------------------------------
mkdir -p "$WORKDIR"
cd "$WORKDIR"

aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "shape_sphere_atlas/*" \
    --exclude "checkpoints*"

# ---------- Pull existing block data (resume) ----------------------------
aws s3 sync "s3://$S3_BUCKET/shape_sphere_atlas/" "$WORKDIR/shape_sphere_atlas/" || true

# ---------- Background log sync ------------------------------------------
sync_logs() {
    while true; do
        sleep 60
        aws s3 cp "$LOG" "s3://$S3_BUCKET/results/${TAG}_live.log" 2>/dev/null || true
        echo "{\"instance\":\"$INSTANCE_ID\",\"tag\":\"$TAG\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"uptime\":\"$(uptime)\"}" \
            | aws s3 cp - "s3://$S3_BUCKET/results/${TAG}_heartbeat.json" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# ---------- Run scan ------------------------------------------------------
echo ""
echo "=================================================================="
echo "=== Starting shape-sphere scan: $TAG ==="
echo "=== $(date -u) ==="
echo "=================================================================="

python3 -u shape_sphere_atlas.py scan \
    --potential "$POTENTIAL" \
    --epsilon "$EPSILON" \
    --grid-theta "$GRID_THETA" \
    --grid-phi "$GRID_PHI" \
    --start-row "$BLOCK_START" \
    --end-row "$BLOCK_END" \
    --workers "$WORKERS" \
    --timeout 180 \
    2>&1 | tee "$WORKDIR/${TAG}.log" &
SCAN_PID=$!
wait $SCAN_PID
EXIT=$?
SCAN_PID=""

echo "=== Scan exit code: $EXIT ==="

# ---------- Final sync (with retries) -------------------------------------
for attempt in 1 2 3; do
    aws s3 sync "$WORKDIR/shape_sphere_atlas/" "s3://$S3_BUCKET/shape_sphere_atlas/" && break
    sleep 10
done
aws s3 cp "$WORKDIR/${TAG}.log" "s3://$S3_BUCKET/results/${TAG}.log" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/${TAG}_full.log" || true

cat > "$WORKDIR/${TAG}_completion.json" <<CEOF
{
    "status": "complete",
    "tag": "$TAG",
    "potential": "$POTENTIAL",
    "block_start": $BLOCK_START,
    "block_end": $BLOCK_END,
    "exit_code": $EXIT,
    "instance": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CEOF
aws s3 cp "$WORKDIR/${TAG}_completion.json" "s3://$S3_BUCKET/results/${TAG}_completion.json" || true

kill $LOG_SYNC_PID 2>/dev/null

echo "=== DONE: $(date -u)  exit=$EXIT ==="
