#!/bin/bash
# Validate survey gravitational results with corrected SymPy
LOG=/var/log/3body-validate-masses.log
exec > "$LOG" 2>&1
set -x
ulimit -s unlimited

S3_BUCKET="3body-compute-290318"
WORKDIR="/opt/3body"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== Survey Mass Validation ==="
echo "Started: $(date -u)"

# Install Python 3.8+ and deps
dnf install -y python3 python3-pip 2>/dev/null || yum install -y python3 python3-pip
amazon-linux-extras install python3.8 -y 2>/dev/null || true
pip3.8 install sympy numpy scipy 2>/dev/null || pip3 install sympy numpy scipy
PYTHON=$(command -v python3.8 || command -v python3)
echo "Python: $($PYTHON --version)"
$PYTHON -c "import sympy; print('SymPy:', sympy.__version__)"

# Pull code
mkdir -p "$WORKDIR"
cd "$WORKDIR"
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKDIR/" \
    --exclude "*.npy" --exclude "*.npz" --exclude "*.pkl" \
    --exclude "atlas_*" --exclude "checkpoints*" --exclude "diagnostic*"

# Background log sync
sync_logs() {
    while true; do
        sleep 30
        aws s3 cp "$LOG" "s3://$S3_BUCKET/results/validate_masses/live.log" 2>/dev/null || true
    done
}
sync_logs &
LOG_SYNC_PID=$!

# Run validation
echo "=== Starting validation ==="
cd "$WORKDIR"
$PYTHON -u validate_survey_masses.py
EXIT_CODE=$?

# Sync results
aws s3 cp validate_survey_results.json \
    "s3://$S3_BUCKET/results/validate_masses/validate_survey_results.json" || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/results/validate_masses/full.log" || true

kill $LOG_SYNC_PID 2>/dev/null
echo "=== DONE: exit=$EXIT_CODE $(date -u) ==="
