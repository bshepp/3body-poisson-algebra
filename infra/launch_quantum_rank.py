#!/usr/bin/env python3
"""
Launch EC2 instance for quantum commutator algebra rank computation.

Computes exact rank over QQ[hbar] using the Moyal bracket [f,g]/(i*hbar)
for N=3, d=2, 1/r potential through level 3.

Usage:
    python launch_quantum_rank.py                    # on-demand r6i.4xlarge
    python launch_quantum_rank.py --spot             # spot (~70% cheaper)
    python launch_quantum_rank.py --dry-run          # preview
"""

import argparse
import base64
import subprocess
import sys
import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__),
                             "userdata_quantum_rank.sh")

AMI = "ami-05024c2628f651b80"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
S3_BUCKET = "3body-compute-290318"

DEFAULT_INSTANCE_TYPE = "r6i.4xlarge"


def sync_code_to_s3(dry_run=False):
    nbody_dir = os.path.join(os.path.dirname(__file__), "..", "nbody")
    nbody_dir = os.path.abspath(nbody_dir)
    s3_dest = f"s3://{S3_BUCKET}/code/nbody/"

    cmd = [
        "aws", "s3", "sync", nbody_dir, s3_dest,
        "--exclude", "*.pyc",
        "--exclude", "__pycache__/*",
        "--exclude", "checkpoints_*/*",
        "--exclude", "*.npy",
        "--exclude", "*.npz",
        "--exclude", "*.pkl",
    ]

    if dry_run:
        print(f"  [DRY RUN] Would sync {nbody_dir} -> {s3_dest}")
        return True

    print(f"  Syncing {nbody_dir} -> {s3_dest} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR syncing code: {result.stderr.strip()}")
        return False
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n")[:20]:
            print(f"    {line}")
    print("  Code sync complete.")
    return True


def launch_instance(instance_type, userdata, spot=False, dry_run=False):
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")
    tag_name = "3body-quantum-rank"

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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return None
    instance_id = result.stdout.strip()
    print(f"  Instance: {instance_id}")
    return instance_id


def main():
    parser = argparse.ArgumentParser(
        description="Launch EC2 for quantum commutator algebra rank")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()

    results_path = "results/quantum_rank/N3_d2_1r"

    rates = {"r6i.4xlarge": 1.008, "r6i.8xlarge": 2.016}
    hourly = rates.get(args.instance_type, 1.0)
    if args.spot:
        hourly *= 0.3

    print("=" * 60)
    print("QUANTUM COMMUTATOR ALGEBRA -- AWS LAUNCHER")
    print("=" * 60)
    print(f"  Task:          N=3 d=2 1/r --quantum --max-level 3")
    print(f"  Bracket:       [f,g]/(i*hbar) (Moyal/Weyl)")
    print(f"  Domain:        QQ[hbar]")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Market:        {'SPOT' if args.spot else 'on-demand'}")
    print(f"  Est. cost:     ~${hourly:.2f}/hr")

    if not args.skip_sync:
        print(f"\n--- Step 1: Sync code to S3 ---")
        if not sync_code_to_s3(dry_run=args.dry_run):
            sys.exit(1)
    else:
        print(f"\n--- Step 1: Skipping code sync ---")

    print(f"\n--- Step 2: Build userdata ---")
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        userdata = f.read()
    print(f"  Userdata: {len(userdata)} bytes")

    print(f"\n--- Step 3: Launch instance ---")
    instance_id = launch_instance(
        args.instance_type, userdata,
        spot=args.spot, dry_run=args.dry_run)

    if instance_id:
        print(f"\n  Monitor:")
        print(f"    aws s3 cp s3://{S3_BUCKET}/{results_path}/heartbeat.json -")
        print(f"    aws s3 cp s3://{S3_BUCKET}/{results_path}/live.log -")
        print(f"\n  Results will be at:")
        print(f"    s3://{S3_BUCKET}/{results_path}/")
        print(f"    s3://{S3_BUCKET}/results/symbolic_rank/"
              f"rank_N3_d2_quantum_1r.json")


if __name__ == "__main__":
    main()
