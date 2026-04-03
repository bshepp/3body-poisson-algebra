#!/usr/bin/env python3
"""
Launch EC2 instance for N-body scaling probe.

Steps:
  1. Sync nbody/ code to S3
  2. Build userdata from template
  3. Launch EC2 instance (on-demand or spot)

Usage:
    python launch_nbody_scaling.py                          # default: N=5..12, on-demand
    python launch_nbody_scaling.py --n-start 9 --n-max 12   # just N=9..12
    python launch_nbody_scaling.py --spot                    # spot instance (~70% cheaper)
    python launch_nbody_scaling.py --dry-run                 # preview without launching
    python launch_nbody_scaling.py --instance-type r6i.8xlarge  # bigger instance
"""

import argparse
import base64
import subprocess
import sys
import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "userdata_nbody_scaling.sh")

# Same infrastructure as existing atlas launches
AMI = "ami-05024c2628f651b80"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
S3_BUCKET = "3body-compute-290318"

DEFAULT_INSTANCE_TYPE = "r6i.4xlarge"  # 16 vCPU, 128GB RAM, ~$1.01/hr


def sync_code_to_s3(dry_run=False):
    """Upload the nbody/ source code to S3."""
    nbody_dir = os.path.join(os.path.dirname(__file__), "nbody")
    s3_dest = f"s3://{S3_BUCKET}/code/nbody/"

    cmd = [
        "aws", "s3", "sync", nbody_dir, s3_dest,
        "--exclude", "*.pyc",
        "--exclude", "__pycache__/*",
        "--exclude", "checkpoints_*/*",
        "--exclude", "*.json",
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
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    print("  Code sync complete.")
    return True


def make_userdata(scaling_args):
    """Read template and substitute __SCALING_ARGS__."""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("__SCALING_ARGS__", scaling_args)


def launch_instance(instance_type, userdata, spot=False, dry_run=False):
    """Launch EC2 instance with the given userdata."""
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")
    tag_name = "3body-nbody-scaling"

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


def estimate_cost(instance_type, spot):
    """Rough cost estimate."""
    # On-demand hourly rates (us-east-1, approximate)
    rates = {
        "r6i.4xlarge": 1.008,
        "r6i.8xlarge": 2.016,
        "c6i.4xlarge": 0.68,
        "c6i.8xlarge": 1.36,
    }
    rate = rates.get(instance_type, 1.0)
    if spot:
        rate *= 0.3  # spot is roughly 30% of on-demand
    # Estimate 2-6 hours depending on N range
    return rate, rate * 2, rate * 6


def main():
    parser = argparse.ArgumentParser(
        description="Launch EC2 for N-body scaling probe")
    parser.add_argument("--n-start", type=int, default=5)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from S3 checkpoints (skip completed N)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--spot", action="store_true",
                        help="Use spot instance (~70%% cheaper)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip code sync to S3 (if already done)")
    args = parser.parse_args()

    scaling_args = (f"--n-start {args.n_start} --n-max {args.n_max} "
                    f"--max-level {args.max_level} --n-samples {args.n_samples}")
    if args.resume:
        scaling_args += " --resume"

    print("=" * 60)
    print("N-BODY SCALING PROBE -- AWS LAUNCHER")
    print("=" * 60)
    print(f"  N range:       {args.n_start} .. {args.n_max}")
    print(f"  Max level:     {args.max_level}")
    print(f"  Samples:       {args.n_samples}")
    print(f"  Resume:        {args.resume}")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Market:        {'SPOT' if args.spot else 'on-demand'}")
    print(f"  Scaling args:  {scaling_args}")

    hourly, est_2h, est_6h = estimate_cost(args.instance_type, args.spot)
    print(f"\n  Est. cost: ~${hourly:.2f}/hr, "
          f"${est_2h:.2f} (2hr) - ${est_6h:.2f} (6hr)")

    # Step 1: Sync code
    if not args.skip_sync:
        print(f"\n--- Step 1: Sync code to S3 ---")
        if not sync_code_to_s3(dry_run=args.dry_run):
            sys.exit(1)
    else:
        print(f"\n--- Step 1: Skipping code sync ---")

    # Step 2: Build userdata
    print(f"\n--- Step 2: Build userdata ---")
    userdata = make_userdata(scaling_args)
    print(f"  Userdata: {len(userdata)} bytes")

    # Step 3: Launch
    print(f"\n--- Step 3: Launch instance ---")
    instance_id = launch_instance(
        args.instance_type, userdata,
        spot=args.spot, dry_run=args.dry_run,
    )

    if instance_id:
        print(f"\n  Monitor:")
        print(f"    aws s3 cp s3://{S3_BUCKET}/results/nbody_scaling/heartbeat.json -")
        print(f"    aws s3 cp s3://{S3_BUCKET}/results/nbody_scaling/live.log -")
        print(f"    aws s3 ls s3://{S3_BUCKET}/results/nbody_scaling/")
        print(f"\n  Results will be at:")
        print(f"    s3://{S3_BUCKET}/results/nbody_scaling/n_body_scaling_results_aws.json")
        print(f"\n  Pull results:")
        print(f"    aws s3 sync s3://{S3_BUCKET}/results/nbody_scaling/ ./nbody_scaling_results/")


if __name__ == "__main__":
    main()
