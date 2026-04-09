#!/bin/bash
exec > /var/log/3body-setup.log 2>&1
set -x

echo "=== 3-Body Level 4 — CSE-Optimised Pipeline ==="
echo "Started: $(date)"

# Install Python 3 and pip
dnf install -y python3 python3-pip

# Install dependencies
pip3 install sympy numpy scipy matplotlib mpmath

echo "Python version: $(python3 --version)"
python3 -c "import sympy; print('SymPy:', sympy.__version__)"

# Create working directory
mkdir -p /opt/3body/checkpoints /opt/3body/results
cd /opt/3body

# Pull code and checkpoints from S3
aws s3 sync s3://3body-compute-290318/code/ /opt/3body/
aws s3 sync s3://3body-compute-290318/checkpoints/ /opt/3body/checkpoints/

echo "Files:"
ls -la /opt/3body/
ls -la /opt/3body/checkpoints/

echo "=== Setup complete, starting computation ==="
echo "Started computation: $(date)"

# Determine workers
NCPU=$(nproc)
WORKERS=$((NCPU - 1))
echo "Using $WORKERS workers out of $NCPU cores"

# Run the NUMERICAL pipeline (derivatives → evaluate → SVD)
# This does everything in one shot: no separate analyse step needed
python3 -u aws_level4.py compute --workers $WORKERS --samples 3000 2>&1 | tee /opt/3body/compute.log

RESULT=$?
echo "=== Computation exit code: $RESULT ==="
echo "Finished: $(date)"

# Upload results to S3
aws s3 cp /opt/3body/compute.log s3://3body-compute-290318/results/compute.log || true
aws s3 sync /opt/3body/checkpoints/ s3://3body-compute-290318/checkpoints/ || true
aws s3 sync /opt/3body/results/ s3://3body-compute-290318/results/ || true

for img in /opt/3body/*.png /opt/3body/results/*.png; do
  [ -f "$img" ] && aws s3 cp "$img" s3://3body-compute-290318/results/ || true
done

echo "=== ALL DONE: $(date) ==="
echo "COMPUTATION_COMPLETE" > /opt/3body/status.txt
aws s3 cp /opt/3body/status.txt s3://3body-compute-290318/status.txt
