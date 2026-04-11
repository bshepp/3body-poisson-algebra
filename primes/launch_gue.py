#!/usr/bin/env python3
"""
Launch EC2 instance for GUE log-gas Poisson algebra computation.

Steps:
  1. Sync nbody/ and primes/ code to S3
  2. Build userdata from template
  3. Launch EC2 instance (spot by default — this is a small job)

Usage:
    python primes/launch_gue.py                    # default: level 3, spot
    python primes/launch_gue.py --dry-run           # preview without launching
    python primes/launch_gue.py --max-level 2       # quick level-2 test
    python primes/launch_gue.py --on-demand         # on-demand instead of spot
"""

import argparse
import base64
import subprocess
import sys
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "userdata_gue.sh")

# Same infrastructure as existing atlas/scaling launches
AMI = "ami-05024c2628f651b80"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
S3_BUCKET = "3body-compute-290318"

# Small instance — N=3 d=1 is lightweight
DEFAULT_INSTANCE_TYPE = "r6i.2xlarge"  # 8 vCPU, 64GB RAM, ~$0.504/hr


def sync_code_to_s3(dry_run=False):
    """Upload nbody/ and primes/ source code to S3."""
    syncs = [
        (os.path.join(PARENT_DIR, "nbody"), f"s3://{S3_BUCKET}/code/nbody/"),
        (SCRIPT_DIR, f"s3://{S3_BUCKET}/code/primes/"),
    ]

    exclude_flags = [
        "--exclude", "*.pyc",
        "--exclude", "__pycache__/*",
        "--exclude", "checkpoints_*/*",
        "--exclude", "*.npy",
        "--exclude", "*.npz",
        "--exclude", "*.pkl",
        "--exclude", "results/*",
    ]

    for local_dir, s3_dest in syncs:
        cmd = ["aws", "s3", "sync", local_dir, s3_dest] + exclude_flags

        if dry_run:
            print(f"  [DRY RUN] Would sync {local_dir} -> {s3_dest}")
            continue

        print(f"  Syncing {local_dir} -> {s3_dest} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR syncing: {result.stderr.strip()}")
            return False

        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")

    print("  Code sync complete.")
    return True


def make_userdata(gue_args, job_name):
    """Read template and substitute placeholders."""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    ud = template.replace("__GUE_ARGS__", gue_args)
    ud = ud.replace("__JOB_NAME__", job_name)
    return ud


def launch_instance(instance_type, userdata, job_name, spot=True, dry_run=False):
    """Launch EC2 instance with the given userdata."""
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")

    cmd = [
        "aws", "ec2", "run-instances",
        "--image-id", AMI,
        "--instance-type", instance_type,
        "--key-name", KEY_NAME,
        "--security-group-ids", SECURITY_GROUP,
        "--iam-instance-profile", f"Name={IAM_PROFILE}",
        "--user-data", ud_b64,
        "--tag-specifications",
        f"ResourceType=instance,Tags=[{{Key=Name,Value={job_name}}}]",
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
        print(f"    Tag:      {job_name}")
        print(f"    Type:     {instance_type}")
        print(f"    Market:   {market}")
        print(f"    AMI:      {AMI}")
        print(f"    Userdata: {len(userdata)} bytes")
        return None

    print(f"\n  Launching {job_name} ({instance_type}, {market})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return None
    instance_id = result.stdout.strip()
    print(f"  Instance: {instance_id}")
    return instance_id


def main():
    parser = argparse.ArgumentParser(
        description="Launch EC2 for GUE log-gas computation")
    parser.add_argument("--max-level", type=int, default=3,
                        help="Maximum bracket level (default: 3)")
    parser.add_argument("--samples", type=int, default=500,
                        help="Phase-space samples (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (default: all)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use on-demand instead of spot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without launching")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip code sync to S3")
    args = parser.parse_args()

    gue_args = f"--max-level {args.max_level} --samples {args.samples} --seed {args.seed}"
    if args.configs:
        gue_args += f" --configs {args.configs}"

    spot = not args.on_demand
    job_name = "3body-gue-logas"

    # Rough cost estimate
    rate = 0.504  # r6i.2xlarge on-demand
    if spot:
        rate *= 0.3
    est_1h = rate

    print("=" * 60)
    print("GUE LOG-GAS -- AWS LAUNCHER")
    print("=" * 60)
    print(f"  Max level:     {args.max_level}")
    print(f"  Samples:       {args.samples}")
    print(f"  Seed:          {args.seed}")
    print(f"  Configs:       {args.configs or 'all (4 configs)'}")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Market:        {'SPOT' if spot else 'on-demand'}")
    print(f"  GUE args:      {gue_args}")
    print(f"\n  Est. cost: ~${est_1h:.2f}/hr")
    print(f"  Expected runtime: ~1 hr for level 3, ~5 min for level 2")

    # Step 1: Sync code
    if not args.skip_sync:
        print(f"\n--- Step 1: Sync code to S3 ---")
        if not sync_code_to_s3(dry_run=args.dry_run):
            sys.exit(1)
    else:
        print(f"\n--- Step 1: Skipping code sync ---")

    # Step 2: Build userdata
    print(f"\n--- Step 2: Build userdata ---")
    userdata = make_userdata(gue_args, job_name)
    print(f"  Userdata: {len(userdata)} bytes")

    if args.dry_run:
        print(f"\n--- Userdata preview (first 40 lines) ---")
        for i, line in enumerate(userdata.split("\n")[:40]):
            print(f"  {line}")
        print("  ...")

    # Step 3: Launch
    print(f"\n--- Step 3: Launch instance ---")
    instance_id = launch_instance(
        args.instance_type, userdata, job_name,
        spot=spot, dry_run=args.dry_run,
    )

    if instance_id:
        print(f"\n  Monitor:")
        print(f"    aws s3 cp s3://{S3_BUCKET}/results/primes/live.log -")
        print(f"    aws s3 cp s3://{S3_BUCKET}/results/primes/instance_heartbeat.json -")
        print(f"    aws s3 ls s3://{S3_BUCKET}/results/primes/")
        print(f"\n  Check completion:")
        print(f"    aws s3 ls s3://{S3_BUCKET}/results/primes/aws_completion.json")
        print(f"\n  Pull results:")
        print(f"    aws s3 sync s3://{S3_BUCKET}/results/primes/ primes/results/")


if __name__ == "__main__":
    main()
