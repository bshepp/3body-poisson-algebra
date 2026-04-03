#!/usr/bin/env python3
"""Launch c6i.4xlarge for plain 1/r^2 atlas (100x100, equal masses/charges, 16 workers)."""
import base64, json, subprocess, sys

template = open('userdata_full_atlas.sh', 'r').read()
job_name = 'atlas-1r2-equal'
atlas_args = '--resolution 100 --potential 1/r^2 --samples 400 --workers 16'
userdata = template.replace('__ATLAS_ARGS__', atlas_args).replace('__JOB_NAME__', job_name)
ud_b64 = base64.b64encode(userdata.encode('utf-8')).decode('ascii')

tag_name = '3body-' + job_name
instance_type = 'c6i.4xlarge'

cmd = [
    'aws', 'ec2', 'run-instances',
    '--image-id', 'ami-05024c2628f651b80',
    '--instance-type', instance_type,
    '--key-name', '3body-compute',
    '--security-group-ids', 'sg-01db2d9932427a00a',
    '--iam-instance-profile', 'Name=3body-profile',
    '--user-data', ud_b64,
    '--tag-specifications',
    'ResourceType=instance,Tags=[{Key=Name,Value=' + tag_name + '}]',
    '--instance-initiated-shutdown-behavior', 'terminate',
    '--query', 'Instances[0].InstanceId',
    '--output', 'text',
]

print('Launching %s (%s, ON-DEMAND)...' % (tag_name, instance_type))
print('Args: %s' % atlas_args)
print('Est. cost: ~2hrs x $0.68/hr = ~$1.36')
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print('ERROR: %s' % result.stderr.strip())
    sys.exit(1)
print('Instance: %s' % result.stdout.strip())
