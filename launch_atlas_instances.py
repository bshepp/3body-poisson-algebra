#!/usr/bin/env python3
"""
Launch AWS EC2 instances for full atlas scans.

Reads the userdata_full_atlas.sh template and substitutes
__ATLAS_ARGS__ and __JOB_NAME__ for each configuration,
then launches on-demand instances.

Usage:
    python launch_atlas_instances.py --tier 1
    python launch_atlas_instances.py --tier 2
    python launch_atlas_instances.py --tier 3
    python launch_atlas_instances.py --tier 4
    python launch_atlas_instances.py --tier all
    python launch_atlas_instances.py --dry-run --tier 1
"""

import argparse
import base64
import subprocess
import sys
import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "userdata_full_atlas.sh")

AMI = "ami-05024c2628f651b80"
KEY_NAME = "3body-compute"
SECURITY_GROUP = "sg-01db2d9932427a00a"
IAM_PROFILE = "3body-profile"
INSTANCE_TYPE_NORMAL = "r6i.4xlarge"
INSTANCE_TYPE_LARGE = "r6i.8xlarge"

TIER_1 = [
    {"job": "atlas-1r3",     "args": "--resolution 100 --potential 1/r^3 --samples 400"},
    {"job": "atlas-log",     "args": "--resolution 100 --potential log --samples 400"},
]

TIER_2 = [
    {"job": "atlas-q1m1m1",   "args": "--resolution 100 --potential 1/r --charges 1 -1 -1 --samples 400"},
    {"job": "atlas-q2m1m1",   "args": "--resolution 100 --potential 1/r --charges 2 -1 -1 --samples 400"},
    {"job": "atlas-q3m1m1",   "args": "--resolution 100 --potential 1/r --charges 3 -1 -1 --samples 400"},
    {"job": "atlas-q1p1m1",   "args": "--resolution 100 --potential 1/r --charges 1 1 -1 --samples 400"},
    {"job": "atlas-q1p1p1",   "args": "--resolution 100 --potential 1/r --charges 1 1 1 --samples 400"},
    {"job": "atlas-1r2-q2m1m1", "args": "--resolution 100 --potential 1/r^2 --charges 2 -1 -1 --samples 400"},
    {"job": "atlas-1r3-q2m1m1", "args": "--resolution 100 --potential 1/r^3 --charges 2 -1 -1 --samples 400"},
]

TIER_3 = [
    {"job": "atlas-binary-star",  "args": "--resolution 100 --scenario binary_star_planet --samples 400"},
    {"job": "atlas-triple-bh",    "args": "--resolution 100 --scenario triple_bh_lisa --samples 400"},
    {"job": "atlas-sun-earth",    "args": "--resolution 100 --scenario sun_earth_moon --samples 800"},
    {"job": "atlas-sun-jup",      "args": "--resolution 100 --scenario sun_jupiter_asteroid --samples 800"},
    {"job": "atlas-positronium",  "args": "--resolution 100 --scenario positronium_neg --samples 400"},
    {"job": "atlas-h-minus",      "args": "--resolution 100 --scenario h_minus_ion --samples 400"},
    {"job": "atlas-lithium",      "args": "--resolution 100 --scenario lithium_ion --samples 400"},
    {"job": "atlas-muonic-he",    "args": "--resolution 100 --scenario muonic_helium --samples 400"},
    {"job": "atlas-h2plus",       "args": "--resolution 100 --scenario h2_plus_ion --samples 400"},
]

TIER_4 = [
    {"job": "atlas-tritium",   "args": "--resolution 100 --scenario tritium_he3 --samples 400",
     "instance_type": INSTANCE_TYPE_LARGE},
    {"job": "atlas-dusty",     "args": "--resolution 100 --scenario dusty_plasma --samples 400",
     "instance_type": INSTANCE_TYPE_LARGE},
]

TIERS = {"1": TIER_1, "2": TIER_2, "3": TIER_3, "4": TIER_4}


def make_userdata(template: str, job_name: str, atlas_args: str) -> str:
    ud = template.replace("__ATLAS_ARGS__", atlas_args)
    ud = ud.replace("__JOB_NAME__", job_name)
    return ud


def launch_instance(job_name, atlas_args, instance_type, dry_run=False,
                    use_spot=False):
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    userdata = make_userdata(template, job_name, atlas_args)
    ud_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")

    tag_name = f"3body-{job_name}"

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
        print(f"    Type: {instance_type}")
        print(f"    Args: {atlas_args}")
        print(f"    Userdata: {len(userdata)} bytes")
        return None

    market = "SPOT" if use_spot else "on-demand"
    print(f"  Launching {tag_name} ({instance_type}, {market})...")
    print(f"    Args: {atlas_args}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
        return None
    instance_id = result.stdout.strip()
    print(f"    Instance: {instance_id}")
    return instance_id


def main():
    parser = argparse.ArgumentParser(description="Launch atlas scan EC2 instances")
    parser.add_argument("--tier", required=True,
                        choices=["1", "2", "3", "4", "all"],
                        help="Which tier to launch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be launched without doing it")
    parser.add_argument("--spot", action="store_true",
                        help="Launch as spot instances (cheaper, may be reclaimed)")
    parser.add_argument("--jobs", nargs="+", default=None,
                        help="Launch only specific jobs by name (e.g. atlas-sun-earth)")
    args = parser.parse_args()

    if args.tier == "all":
        tiers_to_run = ["1", "2", "3", "4"]
    else:
        tiers_to_run = [args.tier]

    for tier_num in tiers_to_run:
        configs = TIERS[tier_num]
        if args.jobs:
            configs = [c for c in configs if c["job"] in args.jobs]
            if not configs:
                continue
        print(f"\n=== Tier {tier_num}: {len(configs)} configurations ===")
        launched = []
        for cfg in configs:
            itype = cfg.get("instance_type", INSTANCE_TYPE_NORMAL)
            iid = launch_instance(cfg["job"], cfg["args"], itype,
                                  dry_run=args.dry_run,
                                  use_spot=args.spot)
            if iid:
                launched.append({"job": cfg["job"], "instance_id": iid})

        if launched and not args.dry_run:
            print(f"\n  Tier {tier_num} launched: {len(launched)} instances")
            for item in launched:
                print(f"    {item['job']}: {item['instance_id']}")


if __name__ == "__main__":
    main()
