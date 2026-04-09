#!/bin/bash
set -ex

START_ROW=__START_ROW__
END_ROW=__END_ROW__

echo "=== 3-Body 1000x1000 Atlas Block: rows $START_ROW - $((END_ROW - 1)) ==="
echo "Started: $(date)"

dnf install -y python3-pip python3-devel gcc
pip3 install sympy numpy

echo "Python version: $(python3 --version)"

mkdir -p /opt/3body
cd /opt/3body

export S3_BUCKET="3body-compute-290318"
aws s3 cp s3://$S3_BUCKET/code/exact_growth.py .
aws s3 cp s3://$S3_BUCKET/code/stability_atlas.py .
aws s3 cp s3://$S3_BUCKET/code/atlas_1000.py .

# Pull any existing checkpoint for this block
aws s3 sync s3://$S3_BUCKET/atlas_1000/ atlas_1000/ || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export S3_BUCKET="3body-compute-290318"

python3 -u atlas_1000.py scan \
  --potential "1/r" --epsilon 5e-3 --grid-n 1000 \
  --start-row $START_ROW --end-row $END_ROW \
  --workers 15 --timeout 600 > atlas_1000.log 2>&1 &
SCAN_PID=$!

# Background log uploader
while kill -0 $SCAN_PID 2>/dev/null; do
    sleep 300
    aws s3 cp /opt/3body/atlas_1000.log \
      s3://$S3_BUCKET/results/atlas_1000_${START_ROW}.log 2>/dev/null || true
done

# Final log upload + shutdown
aws s3 cp /opt/3body/atlas_1000.log \
  s3://$S3_BUCKET/results/atlas_1000_${START_ROW}.log 2>/dev/null || true
echo "Scan complete. Shutting down at $(date)"
sudo shutdown -h now
