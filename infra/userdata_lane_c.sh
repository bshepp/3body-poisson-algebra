#!/bin/bash
exec > /var/log/3body-lane-c.log 2>&1
set -x

echo "=== 3-Body Lane C: N=3, d=2, V=1/r, level 4 mod-p ==="
echo "Started: $(date)"

# Toolchain.  python-flint 0.8.0 wheels need Python >= 3.10, so we pull
# python3.11 from AL2023's repo.  AMI must be Amazon Linux 2023 — the older
# AL2 base only offers python3.7 in extras and pip will silently reject
# every modern sympy/numpy/python-flint version (this is the bug the
# previous Lane C run hit).
dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++
PYTHON=$(command -v python3.11)
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install sympy==1.13.3 numpy mpmath python-flint==0.8.0

$PYTHON -c "import sympy, flint, numpy, mpmath; print('SymPy', sympy.__version__, '/ FLINT', flint.__version__, '/ NumPy', numpy.__version__, '/ mpmath', mpmath.__version__)"

# Layout mirrors aws_level4 pattern
mkdir -p /opt/3body/lane_c
cd /opt/3body

export S3_BUCKET="3body-compute-290318"
export S3_PREFIX="lane_c"
export WORK_DIR="/opt/3body/lane_c"
export MAX_WALLTIME_S="64800"   # 18h hard budget; spot reclaim still safe via periodic sync
export BATCH_SAVE="25"

# Pull code (engine + driver)
aws s3 sync s3://${S3_BUCKET}/code/ /opt/3body/ --no-progress
# Resume from any prior checkpoint
aws s3 sync s3://${S3_BUCKET}/${S3_PREFIX}/checkpoints/ ${WORK_DIR}/ --no-progress || true

ls -la /opt/3body/
ls -la /opt/3body/nbody/
ls -la ${WORK_DIR}/

echo "=== Starting lane_c_aws_driver.py ==="
set +e   # we want to capture the exit code, not abort the post-run sync
$PYTHON -u /opt/3body/bench_flint/lane_c_aws_driver.py \
    --max-level 4 --n-samples 120 \
    2>&1 | tee /opt/3body/lane_c_run.log
# tee returns 0 even when the driver fails — read the driver's actual code
RESULT=${PIPESTATUS[0]}
set -e
echo "=== Driver exit code: $RESULT ==="
echo "Finished: $(date)"

# Final hand-over: sync everything (driver already does this in `finally`)
aws s3 sync ${WORK_DIR}/ s3://${S3_BUCKET}/${S3_PREFIX}/checkpoints/ --no-progress || true
aws s3 cp /opt/3body/lane_c_run.log s3://${S3_BUCKET}/${S3_PREFIX}/lane_c_run.log || true
aws s3 cp /var/log/3body-lane-c.log s3://${S3_BUCKET}/${S3_PREFIX}/3body-lane-c.log || true

if [ "$RESULT" -eq 0 ]; then
    echo "LANE_C_DONE" > /opt/3body/status_lane_c.txt
else
    echo "LANE_C_FAILED exit=$RESULT" > /opt/3body/status_lane_c.txt
fi
aws s3 cp /opt/3body/status_lane_c.txt s3://${S3_BUCKET}/${S3_PREFIX}/status_lane_c.txt

# Self-terminate via shutdown -h (instance launched with shutdown-behavior=terminate)
shutdown -h +2
echo "=== ALL DONE: $(date) ==="
