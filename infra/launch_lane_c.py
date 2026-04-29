#!/usr/bin/env python3
"""
Launch a single AWS spot instance for Lane C (N=3, d=2, V=1/r, level 4
mod-p rank). Mirrors the contract used by `launch_atlas_instances.py`.

Pre-flight (one-time, do this manually, NOT here):
    aws s3 sync ./bench_flint        s3://3body-compute-290318/code/bench_flint/ --exclude 'venv/*' --exclude '_*'
    aws s3 sync ./nbody              s3://3body-compute-290318/code/nbody/
    aws s3 cp   ./exact_growth.py    s3://3body-compute-290318/code/exact_growth.py
    aws s3 cp   ./exact_growth_cm.py s3://3body-compute-290318/code/exact_growth_cm.py

Usage:
    python launch_lane_c.py --dry-run
    python launch_lane_c.py            # spot launch (default)
    python launch_lane_c.py --on-demand
"""
import argparse
import base64
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "userdata_lane_c.sh")

# AL2023 latest x86_64.  The shared AL2 AMI used by older lanes ships
# Python 3.7, which silently breaks sympy>=1.13 / python-flint installs.
# `resolve:ssm:` lets EC2 dereference the parameter at launch time.
AMI = "resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
INSTANCE_TYPE = "r6a.16xlarge"   # 64 vCPU / 512 GiB — matches the lane plan
JOB_NAME = "lane-c-l4-1r"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--on-demand", action="store_true",
                    help="Use on-demand instead of spot (default: spot).")
    args = ap.parse_args()

    with open(TEMPLATE, "r", encoding="utf-8") as f:
        userdata = f.read()
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")

    cmd = [
        "aws", "ec2", "run-instances",
        "--image-id", AMI,
        "--instance-type", INSTANCE_TYPE,
        "--key-name", KEY_NAME,
        "--security-group-ids", SECURITY_GROUP,
        "--iam-instance-profile", f"Name={IAM_PROFILE}",
        "--user-data", ud_b64,
        "--tag-specifications",
        f"ResourceType=instance,Tags=[{{Key=Name,Value=3body-{JOB_NAME}}}]",
        "--instance-initiated-shutdown-behavior", "terminate",
        "--block-device-mappings",
        json.dumps([{
            "DeviceName": "/dev/xvda",
            "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True},
        }]),
        "--query", "Instances[0].InstanceId",
        "--output", "text",
    ]

    if not args.on_demand:
        cmd.extend([
            "--instance-market-options",
            json.dumps({
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }),
        ])

    if args.dry_run:
        market = "on-demand" if args.on_demand else "SPOT"
        print(f"[DRY RUN] would launch {INSTANCE_TYPE} ({market}) for {JOB_NAME}")
        print(f"  userdata = {len(userdata)} bytes")
        print(f"  template = {TEMPLATE}")
        return

    market = "on-demand" if args.on_demand else "SPOT"
    print(f"Launching {INSTANCE_TYPE} ({market}) for {JOB_NAME}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: {res.stderr.strip()}")
        raise SystemExit(1)
    iid = res.stdout.strip()
    print(f"Instance: {iid}")
    print(f"Watch:    aws ec2 describe-instances --instance-ids {iid} "
          f"--query 'Reservations[0].Instances[0].State.Name' --output text")
    print(f"Logs:     aws s3 ls s3://3body-compute-290318/lane_c/")


if __name__ == "__main__":
    main()
