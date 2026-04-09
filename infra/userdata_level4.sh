#!/bin/bash
exec > /var/log/3body-level4.log 2>&1
set -x

echo "=== 3-Body Level 4 Multi-Configuration Pipeline ==="
echo "Started: $(date)"

dnf install -y python3 python3-pip
pip3 install sympy numpy scipy matplotlib

echo "Python version: $(python3 --version)"
python3 -c "import sympy; print('SymPy:', sympy.__version__)"

mkdir -p /opt/3body/checkpoints /opt/3body/results
cd /opt/3body

aws s3 sync s3://3body-compute-290318/code/ /opt/3body/
aws s3 sync s3://3body-compute-290318/checkpoints/ /opt/3body/checkpoints/

echo "Code files:"
ls -la /opt/3body/
echo "Checkpoints:"
ls -la /opt/3body/checkpoints/

export S3_BUCKET="3body-compute-290318"

echo "=== Starting Level 4 batch computation ==="
echo "Started computation: $(date)"

python3 -u aws_level4.py compute --batch 2>&1 | tee /opt/3body/level4_batch.log

RESULT=$?
echo "=== Computation exit code: $RESULT ==="
echo "Finished: $(date)"

aws s3 sync /opt/3body/results/ s3://3body-compute-290318/results/ || true
aws s3 sync /opt/3body/checkpoints/ s3://3body-compute-290318/checkpoints/ || true
aws s3 cp /opt/3body/level4_batch.log s3://3body-compute-290318/results/level4_batch.log || true

echo "LEVEL4_COMPLETE" > /opt/3body/status_l4.txt
aws s3 cp /opt/3body/status_l4.txt s3://3body-compute-290318/status_l4.txt

echo "=== ALL DONE: $(date) ==="
