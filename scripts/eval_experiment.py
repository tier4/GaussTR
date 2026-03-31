#!/usr/bin/env python
"""Evaluate a PG-Occ experiment using 5-segment sliding average analysis.

Usage:
    # Evaluate and print report
    python -m scripts.eval_experiment --run-name ar_mar12_001

    # Evaluate and auto-append to results.tsv
    python -m scripts.eval_experiment --run-name ar_mar12_001 --log

    # JSON output (for programmatic use)
    python -m scripts.eval_experiment --run-name ar_mar12_001 --json

    # List all experiments matching a prefix
    python -m scripts.eval_experiment --list ar_mar12

    # Generate next experiment name for today
    python -m scripts.eval_experiment --next-name

Reads W&B metrics, computes 5-segment sliding averages, and outputs a
structured verdict (IMPROVED / REGRESSED / FLAT) for each metric.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_TSV = os.path.join(_project_root, 'results.tsv')

# Setup W&B API key
if not os.environ.get('WANDB_API_KEY'):
    _key_path = os.path.expanduser('~/.wandb_api_key')
    if os.path.exists(_key_path):
        with open(_key_path) as _f:
            os.environ['WANDB_API_KEY'] = _f.read().strip()

import wandb
import yaml


def _load_wandb_config():
    """Load local W&B config from .wandb_config.yaml."""
    cfg_path = os.path.join(_project_root, '.wandb_config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}

_WANDB_CFG = _load_wandb_config()
WANDB_ENTITY = _WANDB_CFG.get('entity', None)
WANDB_PROJECT = _WANDB_CFG.get('default_projects', {}).get('pgocc_t4', 'pgocc-t4')


# Metrics where LOWER is better
LOWER_IS_BETTER = {
    'train_loss', 'train/depth_0', 'train/depth_1', 'train/depth_2',
    'train/ov_cos_0', 'train/ov_cos_1', 'train/ov_cos_2',
    'train/ov_mse_0', 'train/ov_mse_1', 'train/ov_mse_2',
    'train/warp_0', 'train/warp_1', 'train/warp_2',
    'train/depth_gt_0', 'train/depth_gt_1', 'train/depth_gt_2',
    'train/sem_ce_0', 'train/sem_ce_1', 'train/sem_ce_2',
    'train/sem_text_0', 'train/sem_text_1', 'train/sem_text_2',
    'train/dyn_cov', 'train/dyn_depth',
}

# Primary metrics for keep/discard decision
PRIMARY_METRICS = [
    'train/depth_0', 'train/ov_cos_0', 'train/warp_0', 'train/depth_gt_0',
]

# Secondary metrics (informational)
SECONDARY_METRICS = [
    'train/ov_mse_0', 'train/sem_ce_0', 'train/sem_text_0',
    'train/dyn_cov', 'train/dyn_depth',
]

IMPROVEMENT_THRESHOLD = -2.0  # % change needed to count as "improved"
REGRESSION_THRESHOLD = 5.0    # % change to count as "regressed"


def _get_api():
    """Get W&B API client."""
    return wandb.Api()


def _find_runs(name_prefix, project=WANDB_PROJECT, entity=WANDB_ENTITY):
    """Find W&B runs matching a name prefix."""
    api = _get_api()
    # W&B uses MongoDB-style filters
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": f"^{name_prefix}"}},
        order="-created_at",
    )
    return list(runs)


def get_next_experiment_name():
    """Generate next experiment name: ar_MMMDD_NNN."""
    today = datetime.now().strftime('%b%d').lower()  # e.g. mar12
    prefix = f'ar_{today}_'

    # Find highest existing number in results.tsv
    max_num = 0
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                name = row.get('experiment', '')
                if name.startswith(prefix):
                    try:
                        num = int(name[len(prefix):])
                        max_num = max(max_num, num)
                    except ValueError:
                        pass

    # Also check W&B for runs not yet in results.tsv
    try:
        runs = _find_runs(prefix)
        for r in runs:
            name = r.name
            if name.startswith(prefix):
                try:
                    num = int(name[len(prefix):])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
    except Exception:
        pass

    return f'{prefix}{max_num + 1:03d}'


def compute_5seg_trend(values, steps):
    """Compute 5-segment sliding average trend."""
    n = len(values)
    if n < 10:
        return None

    seg_size = n // 5
    segs = []
    seg_steps = []
    for i in range(5):
        start = i * seg_size
        end = start + seg_size if i < 4 else n
        seg_avg = sum(values[start:end]) / (end - start)
        segs.append(seg_avg)
        seg_steps.append((steps[start], steps[end - 1]))

    pct_change = (segs[4] - segs[0]) / abs(segs[0]) * 100 if abs(segs[0]) > 1e-8 else 0
    mid_trend = (segs[2] - segs[0]) / abs(segs[0]) * 100 if abs(segs[0]) > 1e-8 else 0
    improving_segs = sum(1 for i in range(1, 5) if segs[i] < segs[i-1])

    return {
        'segs': segs,
        'seg_steps': seg_steps,
        'pct_change': pct_change,
        'mid_trend': mid_trend,
        'improving_segs': improving_segs,
        'n_points': n,
        'step_range': (steps[0], steps[-1]),
    }


def classify_trend(trend, metric_name):
    """Classify a metric trend as IMPROVED / REGRESSED / FLAT."""
    if trend is None:
        return 'INSUFFICIENT_DATA'

    pct = trend['pct_change']
    lower_better = metric_name in LOWER_IS_BETTER

    if lower_better:
        if pct < IMPROVEMENT_THRESHOLD:
            return 'IMPROVED'
        elif pct > REGRESSION_THRESHOLD:
            return 'REGRESSED'
        else:
            return 'FLAT'
    else:
        if pct > -IMPROVEMENT_THRESHOLD:
            return 'IMPROVED'
        elif pct < -REGRESSION_THRESHOLD:
            return 'REGRESSED'
        else:
            return 'FLAT'


def get_run_metrics(run_name, project=WANDB_PROJECT, entity=WANDB_ENTITY):
    """Fetch all metric histories for a run from W&B."""
    runs = _find_runs(run_name, project, entity)
    if not runs:
        return None, None

    run = runs[0]

    # W&B history() returns a pandas-like iterator of logged metrics
    # Use scan_history for full metric history (no sampling)
    results = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        history = list(run.scan_history(keys=[metric, '_step'], page_size=10000))
        if not history:
            continue
        steps = [h.get('_step', i) for i, h in enumerate(history)]
        vals = [h[metric] for h in history if metric in h]
        steps = steps[:len(vals)]
        if not vals:
            continue
        trend = compute_5seg_trend(vals, steps)
        results[metric] = {
            'trend': trend,
            'verdict': classify_trend(trend, metric),
            'latest': vals[-1] if vals else None,
        }

    # Gather available metrics from summary
    available = sorted(run.summary.keys()) if run.summary else []

    return results, {
        'run_id': run.id,
        'run_name': run.name,
        'status': run.state,
        'available_metrics': available,
        'url': run.url,
    }


def get_git_short_hash():
    """Get current git short hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_project_root, text=True
        ).strip()
    except Exception:
        return 'unknown'


def compute_overall_verdict(results):
    """Compute overall keep/discard verdict."""
    improved = 0
    regressed = 0
    for metric in PRIMARY_METRICS:
        if metric not in results:
            continue
        v = results[metric]['verdict']
        if v == 'IMPROVED':
            improved += 1
        elif v == 'REGRESSED':
            regressed += 1

    if regressed >= 2:
        return 'discard', improved, regressed
    elif improved >= 2 and regressed == 0:
        return 'keep', improved, regressed
    elif improved >= 1 and regressed == 0:
        return 'keep', improved, regressed
    else:
        return 'discard', improved, regressed


def append_to_results_tsv(run_name, results, description=''):
    """Append experiment results to results.tsv."""
    commit = get_git_short_hash()
    status, _, _ = compute_overall_verdict(results)

    # Extract pct_change for primary metrics
    def get_pct(metric):
        if metric in results and results[metric]['trend']:
            return f"{results[metric]['trend']['pct_change']:+.1f}"
        return 'N/A'

    row = {
        'commit': commit,
        'experiment': run_name,
        'depth_0_pct': get_pct('train/depth_0'),
        'ov_cos_pct': get_pct('train/ov_cos_0'),
        'warp_pct': get_pct('train/warp_0'),
        'depth_gt_pct': get_pct('train/depth_gt_0'),
        'status': status,
        'description': description,
    }

    header = ['commit', 'experiment', 'depth_0_pct', 'ov_cos_pct', 'warp_pct', 'depth_gt_pct', 'status', 'description']

    file_exists = os.path.exists(RESULTS_TSV)
    with open(RESULTS_TSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  → Logged to {RESULTS_TSV}")


def print_report(results, run_info, output_json=False):
    """Print human-readable evaluation report."""
    if output_json:
        out = {'run': run_info, 'metrics': {}}
        for m, r in results.items():
            out['metrics'][m] = {
                'verdict': r['verdict'],
                'pct_change': r['trend']['pct_change'] if r['trend'] else None,
                'latest': r['latest'],
            }
        status, imp, reg = compute_overall_verdict(results)
        out['overall'] = {'verdict': status, 'improved': imp, 'regressed': reg}
        print(json.dumps(out, indent=2))
        return status

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT EVALUATION: {run_info['run_name']}")
    print(f"  Status: {run_info['status']}")
    if run_info.get('url'):
        print(f"  W&B URL: {run_info['url']}")
    print(f"{'='*70}\n")

    print("PRIMARY METRICS (決定 keep/discard):")
    print(f"{'Metric':<25} {'Verdict':<15} {'Seg1 → Seg5':>22} {'Change':>10}")
    print("-" * 72)

    for metric in PRIMARY_METRICS:
        if metric not in results:
            print(f"  {metric:<23} {'N/A':<15}")
            continue
        r = results[metric]
        t = r['trend']
        if t:
            seg_str = f"{t['segs'][0]:.4f} -> {t['segs'][4]:.4f}"
            pct_str = f"{t['pct_change']:+.1f}%"
        else:
            seg_str = "insufficient data"
            pct_str = ""
        print(f"  {metric:<23} {r['verdict']:<15} {seg_str:>22} {pct_str:>10}")

    print(f"\nSECONDARY METRICS (参考):")
    print("-" * 72)
    for metric in SECONDARY_METRICS:
        if metric not in results:
            continue
        r = results[metric]
        t = r['trend']
        if t:
            seg_str = f"{t['segs'][0]:.4f} -> {t['segs'][4]:.4f}"
            pct_str = f"{t['pct_change']:+.1f}%"
        else:
            seg_str = "insufficient data"
            pct_str = ""
        print(f"  {metric:<23} {r['verdict']:<15} {seg_str:>22} {pct_str:>10}")

    status, improved, regressed = compute_overall_verdict(results)
    print(f"\n{'='*70}")
    print(f"  OVERALL VERDICT: {status.upper()}")
    print(f"  (improved={improved}, regressed={regressed}, out of {len(PRIMARY_METRICS)} primary)")
    print(f"{'='*70}\n")

    return status


def list_runs(prefix, project=WANDB_PROJECT, entity=WANDB_ENTITY):
    """List all runs matching a prefix."""
    runs = _find_runs(prefix, project, entity)
    if not runs:
        print(f"No runs matching '{prefix}*'")
        return

    print(f"\n{'Run Name':<30} {'Status':<12} {'Steps':>8} {'train_loss':>12} {'URL'}")
    print("-" * 100)
    for r in runs:
        tl = r.summary.get('train_loss', 0) if r.summary else 0
        step_count = r.summary.get('_step', 0) if r.summary else 0
        url = r.url or ''
        print(f"  {r.name:<28} {r.state:<12} {step_count:>8} {tl:>12.4f} {url}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PG-Occ experiment')
    parser.add_argument('--run-name', help='W&B run name (prefix match)')
    parser.add_argument('--project', default=WANDB_PROJECT, help='W&B project name')
    parser.add_argument('--entity', default=WANDB_ENTITY, help='W&B entity/team')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--log', action='store_true', help='Auto-append to results.tsv')
    parser.add_argument('--description', default='', help='Description for results.tsv')
    parser.add_argument('--next-name', action='store_true', help='Print next experiment name and exit')
    parser.add_argument('--list', metavar='PREFIX', help='List all runs matching prefix')
    args = parser.parse_args()

    if args.next_name:
        print(get_next_experiment_name())
        return

    if args.list is not None:
        list_runs(args.list, args.project, args.entity)
        return

    if not args.run_name:
        parser.error("--run-name is required (unless using --next-name or --list)")

    results, run_info = get_run_metrics(args.run_name, args.project, args.entity)
    if results is None:
        print(f"ERROR: No run found matching '{args.run_name}'", file=sys.stderr)
        sys.exit(1)

    verdict = print_report(results, run_info, output_json=args.json)

    if args.log:
        append_to_results_tsv(
            run_info['run_name'], results, description=args.description
        )

    # Exit code: 0 = keep, 1 = discard
    if verdict and 'keep' in str(verdict).lower():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
