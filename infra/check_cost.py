import json, subprocess, sys

def get_cost(start, end):
    r = subprocess.run([
        'aws', 'ce', 'get-cost-and-usage',
        '--time-period', f'Start={start},End={end}',
        '--granularity', 'DAILY',
        '--metrics', 'UnblendedCost',
        '--output', 'json'
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR:", r.stderr)
        return
    d = json.loads(r.stdout)
    for t in d['ResultsByTime']:
        total = t.get('Total', {}).get('UnblendedCost', {}).get('Amount')
        if total is None and 'Results' in t:
            total = t['Results'][0]['Total']['UnblendedCost']['Amount']
        if total is None and 'Groups' in t:
            total = sum(float(g['Metrics']['UnblendedCost']['Amount']) for g in t['Groups'])
        amt = float(total) if total else 0.0
        print(f"  {t['TimePeriod']['Start']}: ${amt:.2f}")
    return d

def get_cost_by_service(start, end):
    r = subprocess.run([
        'aws', 'ce', 'get-cost-and-usage',
        '--time-period', f'Start={start},End={end}',
        '--granularity', 'DAILY',
        '--metrics', 'UnblendedCost',
        '--group-by', 'Type=DIMENSION,Key=SERVICE',
        '--output', 'json'
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR:", r.stderr)
        return
    d = json.loads(r.stdout)
    for t in d['ResultsByTime']:
        print(f"\n  {t['TimePeriod']['Start']}:")
        groups = sorted(t['Groups'], key=lambda g: -float(g['Metrics']['UnblendedCost']['Amount']))
        for g in groups:
            amt = float(g['Metrics']['UnblendedCost']['Amount'])
            if amt > 0.01:
                print(f"    {g['Keys'][0]:50s} ${amt:.2f}")

print("=== Today (March 26) ===")
get_cost('2026-03-26', '2026-03-27')

print("\n=== This week ===")
get_cost('2026-03-24', '2026-03-27')

print("\n=== March total ===")
get_cost('2026-03-01', '2026-03-27')

print("\n=== Today by service ===")
get_cost_by_service('2026-03-26', '2026-03-27')

print("\n=== Last 3 days by service ===")
get_cost_by_service('2026-03-24', '2026-03-27')
