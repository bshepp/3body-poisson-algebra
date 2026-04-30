#!/usr/bin/env python3
"""
Launch EC2 instance for the Q(eps) symbolic-nullspace job.

Computes the exact left-null kernel of the (4,3) binary-collision
collision matrix over Q(eps) (instead of just at sample epsilons).

Usage:
    python launch_qeps.py --dry-run                      # preview only
    python launch_qeps.py                                 # on-demand r6i.4xlarge
    python launch_qeps.py --spot                          # spot (~70% cheaper)
    python launch_qeps.py --instance-type r6i.8xlarge     # more RAM
"""

import argparse
import base64
import subprocess
import sys
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
TEMPLATE_PATH = os.path.join(HERE, "userdata_qeps.sh")

AMI = "ami-05024c2628f651b80"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
S3_BUCKET = "3body-compute-290318"

DEFAULT_INSTANCE_TYPE = "r6i.4xlarge"


def sync_inputs_to_s3(dry_run=False):
    """Upload the three Python files + level_3.pkl checkpoint."""
    items = [
        (os.path.join(REPO, "collision_syzygy_v2.py"),
         f"s3://{S3_BUCKET}/code/3body/collision_syzygy_v2.py"),
        (os.path.join(REPO, "collision_syzygy_v2_qeps.py"),
         f"s3://{S3_BUCKET}/code/3body/collision_syzygy_v2_qeps.py"),
        (os.path.join(REPO, "exact_growth.py"),
         f"s3://{S3_BUCKET}/code/3body/exact_growth.py"),
        (os.path.join(REPO, "checkpoints", "level_3.pkl"),
         f"s3://{S3_BUCKET}/code/3body/checkpoints/level_3.pkl"),
    ]
    for src, dst in items:
        if not os.path.isfile(src):
            print(f"  ERROR: missing {src}")
            return False
        if dry_run:
            print(f"  [DRY] {src}  ->  {dst}")
            continue
        print(f"  uploading {os.path.basename(src)} -> {dst}")
        r = subprocess.run(["aws", "s3", "cp", src, dst],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ERROR: {r.stderr.strip()}")
            return False
    return True


def launch_instance(instance_type, userdata, spot=False, dry_run=False):
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")
    tag_name = "3body-qeps-nullspace"

    cmd = [
        "aws", "ec2", "run-instances",
        "--image-id", AMI,
        "--instance-type", instance_type,
        "--key-name", KEY_NAME,
        "--security-group-ids", SECURITY_GROUP,
        "--iam-instance-profile", f"Name={IAM_PROFILE}",
        "--user-data", ud_b64,
        "--tag-specifications",
        f"ResourceType=instance,Tags=[{{Key=Name,Value={tag_name}}}]",
        "--instance-initiated-shutdown-behavior", "terminate",
        "--query", "Instances[0].InstanceId",
        "--output", "text",
    ]

    if spot:
        spot_opts = json.dumps({
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        })
        cmd.extend(["--instance-market-options", spot_opts])

    market = "SPOT" if spot else "on-demand"

    if dry_run:
        print(f"\n  [DRY RUN] Would launch:")
        print(f"    Tag:      {tag_name}")
        print(f"    Type:     {instance_type}")
        print(f"    Market:   {market}")
        print(f"    Userdata: {len(userdata)} bytes")
        return None

    print(f"\n  Launching {tag_name} ({instance_type}, {market})...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr.strip()}")
        return None
    instance_id = r.stdout.strip()
    print(f"  Instance: {instance_id}")
    return instance_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("Q(eps) symbolic-nullspace job  (collision_syzygy_v2_qeps.py)")
    print("=" * 72)
    print(f"  instance type : {args.instance_type}")
    print(f"  market        : {'spot' if args.spot else 'on-demand'}")
    print(f"  dry-run       : {args.dry_run}")
    print()

    print("[1/3] Syncing inputs (3 .py + level_3.pkl) to S3 ...")
    if not sync_inputs_to_s3(dry_run=args.dry_run):
        sys.exit(1)

    print("\n[2/3] Reading userdata template ...")
    with open(TEMPLATE_PATH, "r") as f:
        userdata = f.read()
    print(f"  {len(userdata)} bytes")

    print("\n[3/3] Launching instance ...")
    iid = launch_instance(args.instance_type, userdata,
                          spot=args.spot, dry_run=args.dry_run)
    if iid:
        print(f"\nDone. Monitor:")
        print(f"  aws s3 cp s3://{S3_BUCKET}/results/collision_syzygy_qeps/live.log -")
        print(f"  aws ec2 describe-instances --instance-ids {iid} \\")
        print(f"      --query 'Reservations[0].Instances[0].State.Name' --output text")
        print(f"\nResult artifact (when complete):")
        print(f"  aws s3 cp s3://{S3_BUCKET}/results/collision_syzygy_qeps/collision_syzygy_qeps.json .")


if __name__ == "__main__":
    main()
