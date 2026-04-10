#!/usr/bin/env python3
"""
Launch EC2 instance for symbolic Gram determinant sweep.

Computes exact Gram matrix G_ij(mu), generator norms ||e_i(mu)||^2,
and det(G(mu)) as rational functions of configuration parameter mu.

Usage:
    python launch_gram_sweep.py                    # on-demand
    python launch_gram_sweep.py --spot             # spot (~70% cheaper)
    python launch_gram_sweep.py --dry-run          # preview
"""

import argparse
import base64
import subprocess
import sys
import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__),
                             "userdata_gram_sweep.sh")

AMI = "ami-05024c2628f651b80"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
S3_BUCKET = "3body-compute-290318"

DEFAULT_INSTANCE_TYPE = "r6i.4xlarge"


def sync_code_to_s3(dry_run=False):
    """Upload nbody/ source code to S3."""
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
        "--exclude", "bell_test_results/*",
        "--exclude", "helium_comparison*/*",
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
    """Launch EC2 instance."""
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")
    tag_name = "3body-gram-sweep"

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
        description="Launch EC2 for symbolic Gram sweep")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--spot", action="store_true",
                        help="Use spot instance")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip code sync to S3")
    args = parser.parse_args()

    results_path = "results/symbolic_gram_sweep"

    rates = {"r6i.4xlarge": 1.008, "r6i.8xlarge": 2.016}
    hourly = rates.get(args.instance_type, 1.0)
    if args.spot:
        hourly *= 0.3

    print("=" * 60)
    print("SYMBOLIC GRAM SWEEP -- AWS LAUNCHER")
    print("=" * 60)
    print(f"  Task:          Gram det & norms for 1/r, r^4, 1/r^4")
    print(f"  Level:         2 (rank 17)")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Market:        {'SPOT' if args.spot else 'on-demand'}")
    print(f"  Est. cost:     ~${hourly:.2f}/hr, "
          f"${hourly*1:.2f} (1hr) - ${hourly*4:.2f} (4hr)")

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
        print(f"    s3://{S3_BUCKET}/{results_path}/1r/")
        print(f"    s3://{S3_BUCKET}/{results_path}/r4/")
        print(f"    s3://{S3_BUCKET}/{results_path}/composite_u4/")
        print(f"\n  Pull results:")
        print(f"    aws s3 sync s3://{S3_BUCKET}/{results_path}/ "
              f"./results/symbolic_gram_sweep/")


if __name__ == "__main__":
    main()
