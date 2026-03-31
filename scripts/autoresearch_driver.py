#!/usr/bin/env python
"""Autoresearch batch driver: run N experiments sequentially, evaluate each, print summary.

Usage:
    python -m scripts.autoresearch_driver --batch experiments.json --gpus 0,1,2,3

experiments.json format:
    [
      {
        "description": "Phase 2 full config",
        "overrides": {
          "model.loss_weights.ov_cos": 2.0,
          "model.loss_weights.branch_cls": 0.5
        }
      },
      ...
    ]

Optional per-experiment fields:
  - "limit_batches": int  (override +trainer.limit_train_batches, default 3000)

Default overrides applied to ALL experiments (merge-with-experiment overrides winning):
  - model.loss_weights.ov_cos=2.0  (current KEEP baseline)

Fixed parameters (always set, not overridable):
  - --config-name pgocc_t4
  - +load_from=/tmp/fixv23_best.ckpt
  - trainer.max_epochs=1
  - data.num_workers=4
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_python = os.path.join(_project_root, '.venv', 'bin', 'python')

# Reuse evaluation helpers from eval_experiment
sys.path.insert(0, _project_root)
from scripts.eval_experiment import (
    get_next_experiment_name,
    get_run_metrics,
    compute_overall_verdict,
    append_to_results_tsv,
    print_report,
    PRIMARY_METRICS,
)

# Default overrides applied to every experiment (experiment overrides take precedence)
DEFAULT_OVERRIDES = {
    'model.loss_weights.ov_cos': 2.0,
}

# Fixed parameters always passed to train_pgocc
FIXED_PARAMS = [
    '--config-name', 'pgocc_t4',
]
FIXED_OVERRIDES = {
    '+load_from': '/tmp/fixv23_best.ckpt',
    'trainer.max_epochs': 1,
    'data.num_workers': 4,
}


def build_command(run_name, gpus, merged_overrides, limit_batches):
    """Build the train_pgocc subprocess command."""
    n_gpus = len(gpus.split(','))
    cmd = [_python, '-m', 'scripts.train_pgocc'] + FIXED_PARAMS + [
        f'run_name={run_name}',
        f'trainer.devices={n_gpus}',
        f'+trainer.limit_train_batches={limit_batches}',
    ]
    for k, v in FIXED_OVERRIDES.items():
        cmd.append(f'{k}={v}')
    for k, v in merged_overrides.items():
        cmd.append(f'{k}={v}')
    return cmd


def run_experiment(run_name, gpus, merged_overrides, limit_batches, log_path):
    """Launch train_pgocc, wait for completion. Returns returncode."""
    cmd = build_command(run_name, gpus, merged_overrides, limit_batches)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus

    print(f"\n[driver] Launching: {' '.join(cmd)}")
    print(f"[driver] Log: {log_path}")

    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                                cwd=_project_root)
    proc.wait()
    return proc.returncode


def evaluate_experiment(run_name, description):
    """Evaluate via W&B and log to results.tsv. Returns (results, verdict_str)."""
    # Wait briefly for W&B to flush final metrics
    time.sleep(5)

    results, run_info = get_run_metrics(run_name)
    if results is None:
        print(f"[driver] WARNING: No W&B run found for '{run_name}'")
        return None, 'ERROR'

    print_report(results, run_info)
    append_to_results_tsv(run_info['run_name'], results, description=description)

    verdict, improved, regressed = compute_overall_verdict(results)
    return results, verdict


def format_pct(results, metric):
    """Format pct_change for summary table."""
    if results and metric in results and results[metric]['trend']:
        pct = results[metric]['trend']['pct_change']
        return f'{pct:+.1f}%'
    return 'N/A'


def check_gpu_availability():
    """Abort if any train_pgocc processes are already running."""
    result = subprocess.run(['pgrep', '-f', 'train_pgocc'], capture_output=True)
    if result.returncode == 0:
        pids = result.stdout.decode().strip().split('\n')
        print(f'[driver] ERROR: train_pgocc already running (PIDs: {", ".join(pids)}). Aborting.')
        sys.exit(1)
    print('[driver] GPU check passed — no conflicting processes.')


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description='Autoresearch batch driver')
    parser.add_argument('--batch', required=True,
                        help='Path to JSON batch file (or - for stdin)')
    parser.add_argument('--gpus', default='0,1,2,3',
                        help='CUDA_VISIBLE_DEVICES (default: 0,1,2,3)')
    args = parser.parse_args()

    # Load batch file
    if args.batch == '-' or args.batch == '/dev/stdin':
        experiments = json.load(sys.stdin)
    else:
        with open(args.batch) as f:
            experiments = json.load(f)

    if not experiments:
        print('[driver] No experiments in batch file. Exiting.')
        sys.exit(0)

    check_gpu_availability()

    print(f'\n[driver] Starting batch of {len(experiments)} experiments on GPUs: {args.gpus}')

    work_dir = os.path.join(_project_root, 'work_dirs', 'pgocc_t4')
    os.makedirs(work_dir, exist_ok=True)

    summary_rows = []

    for i, exp in enumerate(experiments, 1):
        description = exp.get('description', '')
        user_overrides = exp.get('overrides', {})
        limit_batches = exp.get('limit_batches', 3000)

        # Merge: defaults < user overrides
        merged_overrides = {**DEFAULT_OVERRIDES, **user_overrides}

        # Generate unique run name
        run_name = get_next_experiment_name()
        log_path = os.path.join(work_dir, f'{run_name}.log')

        print(f'\n{"="*70}')
        print(f'[driver] Experiment {i}/{len(experiments)}: {run_name}')
        print(f'[driver] Description: {description}')
        print(f'[driver] Overrides: {merged_overrides}')
        print(f'[driver] limit_batches: {limit_batches}')
        print(f'{"="*70}')

        t0 = time.time()
        rc = run_experiment(run_name, args.gpus, merged_overrides, limit_batches, log_path)
        elapsed = time.time() - t0

        if rc != 0:
            print(f'[driver] WARNING: Training exited with code {rc}. Evaluating anyway.')

        results, verdict = evaluate_experiment(run_name, description)

        summary_rows.append({
            'idx': i,
            'name': run_name,
            'description': description,
            'results': results,
            'verdict': verdict,
            'elapsed_min': elapsed / 60,
        })

        print(f'[driver] Experiment {run_name} done in {elapsed/60:.1f} min — verdict: {verdict.upper()}')

    # Final summary table
    print(f'\n\n{"="*80}')
    print('=== AUTORESEARCH BATCH COMPLETE ===')
    print(f'{"="*80}')
    header = f"{'#':>3} | {'Name':<16} | {'depth_0':>9} | {'ov_cos':>8} | {'warp':>8} | {'depth_gt':>10} | {'Verdict':<8} | Description"
    print(header)
    print('-' * len(header))
    for row in summary_rows:
        r = row['results']
        depth_0 = format_pct(r, 'train/depth_0')
        ov_cos  = format_pct(r, 'train/ov_cos_0')
        warp    = format_pct(r, 'train/warp_0')
        depth_gt = format_pct(r, 'train/depth_gt_0')
        verdict = row['verdict'].upper()
        desc = row['description'][:40]
        print(f"{row['idx']:>3} | {row['name']:<16} | {depth_0:>9} | {ov_cos:>8} | {warp:>8} | {depth_gt:>10} | {verdict:<8} | {desc}")

    print(f'{"="*80}')
    n_keep = sum(1 for r in summary_rows if r['verdict'] == 'keep')
    n_discard = sum(1 for r in summary_rows if r['verdict'] == 'discard')
    print(f'KEEP: {n_keep}  DISCARD: {n_discard}  ERROR: {len(summary_rows)-n_keep-n_discard}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
