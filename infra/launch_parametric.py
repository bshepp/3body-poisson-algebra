#!/usr/bin/env python3
"""
Launch AWS EC2 instances for parametric exponent sweep (1/r^n, n in [-5, +5]).

Reads the userdata_parametric_atlas.sh template, substitutes __PARA_ARGS__
and __JOB_NAME__, and launches r6i.4xlarge instances (16 vCPUs) — one per
range segment so all 1,000+ exponents run in ~7-8h.

Usage:
    python launch_parametric.py --dry-run
    python launch_parametric.py
    python launch_parametric.py --spot
    python launch_parametric.py --jobs para-000-to-100 para-101-to-200
"""

import argparse
import base64
import subprocess
import sys
import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "userdata_parametric_atlas.sh")

AMI = "ami-05024c2628f651b80"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
INSTANCE_TYPE = "r6i.4xlarge"   # 16 vCPUs — matches --workers 16

_COMMON = "--resolution 50 --samples 200 --workers 16 --level 3"

PARA_JOBS = [
    {
        "job": "para-neg5-neg201",
        "args": f"--exponent-range -5.0 -2.01 {_COMMON}",
    },
    {
        "job": "para-neg200-neg001",
        "args": f"--exponent-range -2.0 -0.01 {_COMMON}",
    },
    {
        "job": "para-000-to-100",
        "args": f"--exponent-range 0.0 1.0 {_COMMON}",
    },
    {
        "job": "para-101-to-200",
        "args": f"--exponent-range 1.01 2.0 {_COMMON}",
    },
    {
        "job": "para-201-to-500",
        "args": f"--exponent-range 2.01 5.0 {_COMMON}",
    },
    {
        "job": "para-special",
        "args": (
            "--exponents"
            " 3.14159265359 2.71828182846 1.61803398875"
            " 1.41421356237 -3.14159265359 -1.61803398875"
            f" {_COMMON}"
        ),
    },
]


def make_userdata(template: str, job_name: str, para_args: str) -> str:
    ud = template.replace("__PARA_ARGS__", para_args)
    ud = ud.replace("__JOB_NAME__", job_name)
    return ud


def launch_instance(job_name, para_args, dry_run=False, use_spot=False):
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    userdata = make_userdata(template, job_name, para_args)
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")

    tag_name = f"3body-{job_name}"

    cmd = [
        "aws", "ec2", "run-instances",
        "--image-id", AMI,
        "--instance-type", INSTANCE_TYPE,
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

    if use_spot:
        spot_opts = json.dumps({
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        })
        cmd.extend(["--instance-market-options", spot_opts])

    if dry_run:
        print(f"  [DRY RUN] {tag_name} {'(SPOT)' if use_spot else '(on-demand)'}")
        print(f"    Type: {INSTANCE_TYPE}")
        print(f"    Args: {para_args}")
        print(f"    Userdata: {len(userdata)} bytes")
        return None

    market = "SPOT" if use_spot else "on-demand"
    print(f"  Launching {tag_name} ({INSTANCE_TYPE}, {market})...")
    print(f"    Args: {para_args}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
        return None
    instance_id = result.stdout.strip()
    print(f"    Instance: {instance_id}")
    return instance_id


def main():
    parser = argparse.ArgumentParser(
        description="Launch parametric exponent sweep EC2 instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be launched without doing it")
    parser.add_argument("--spot", action="store_true",
                        help="Launch as spot instances (cheaper, SIGTERM handler present)")
    parser.add_argument("--jobs", nargs="+", default=None,
                        help="Launch only specific jobs by name")
    args = parser.parse_args()

    configs = PARA_JOBS
    if args.jobs:
        configs = [c for c in configs if c["job"] in args.jobs]
        if not configs:
            print(f"No matching jobs found. Available: {[c['job'] for c in PARA_JOBS]}")
            sys.exit(1)

    mode = "SPOT" if args.spot else "on-demand"
    print(f"\n=== Parametric sweep: {len(configs)} job(s), {INSTANCE_TYPE}, {mode} ===\n")

    launched = []
    for cfg in configs:
        iid = launch_instance(cfg["job"], cfg["args"],
                              dry_run=args.dry_run,
                              use_spot=args.spot)
        if iid:
            launched.append({"job": cfg["job"], "instance_id": iid})

    if launched and not args.dry_run:
        print(f"\n  Launched: {len(launched)} instances")
        for item in launched:
            print(f"    {item['job']}: {item['instance_id']}")
        print(f"\n  Monitor with:")
        for item in launched:
            job = item["job"]
            print(f"    aws s3 cp s3://3body-compute-290318/results/parametric/{job}/live.log -")


if __name__ == "__main__":
    main()
