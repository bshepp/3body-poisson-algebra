#!/bin/bash
exec > /var/log/3body-multi-eps.log 2>&1
set -x

echo "=== 3-Body Multi-Epsilon Atlas ==="
echo "Started: $(date)"

dnf install -y python3 python3-pip
pip3 install sympy numpy scipy matplotlib plotly

echo "Python version: $(python3 --version)"
python3 -c "import sympy; print('SymPy:', sympy.__version__)"

mkdir -p /opt/3body
cd /opt/3body

aws s3 sync s3://3body-compute-290318/code/ /opt/3body/
aws s3 sync s3://3body-compute-290318/atlas_output_hires/ /opt/3body/atlas_output_hires/

echo "Code files:"
ls -la /opt/3body/
echo "Existing atlas data:"
find /opt/3body/atlas_output_hires/ -name "checkpoint.json" -exec echo {} \; -exec cat {} \;

export S3_BUCKET="3body-compute-290318"

echo "=== Starting multi-epsilon scan ==="
echo "Started computation: $(date)"

python3 -u multi_epsilon_atlas.py all 2>&1 | tee /opt/3body/multi_eps.log

RESULT=$?
echo "=== Computation exit code: $RESULT ==="
echo "Finished: $(date)"

aws s3 sync /opt/3body/atlas_output_hires/ s3://3body-compute-290318/atlas_output_hires/ || true
aws s3 cp /opt/3body/multi_eps.log s3://3body-compute-290318/results/multi_eps.log || true

echo "MULTI_EPS_COMPLETE" > /opt/3body/status.txt
aws s3 cp /opt/3body/status.txt s3://3body-compute-290318/status.txt

echo "=== ALL DONE: $(date) ==="
