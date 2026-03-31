#!/usr/bin/env python
"""Migrate historical MLflow experiment data to Weights & Biases.

Reads all runs from the local MLflow SQLite database, deduplicates them,
and re-creates them as W&B runs with full metric histories.

Cleanup logic:
- Duplicate run names: keep the run with the most data points
- Unnamed runs (None_*) with <10 steps: discard
- Runs with 0 metrics: discard

Usage:
    python -m scripts.migrate_mlflow_to_wandb --dry-run
    python -m scripts.migrate_mlflow_to_wandb
    python -m scripts.migrate_mlflow_to_wandb --experiment-id 13
"""

import argparse
import os
import sys
import time
import yaml

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load W&B credentials
if not os.environ.get('WANDB_API_KEY'):
    _key_path = os.path.expanduser('~/.wandb_api_key')
    if os.path.exists(_key_path):
        with open(_key_path) as f:
            os.environ['WANDB_API_KEY'] = f.read().strip()

# Setup MLflow URI (read-only, for migration source)
mlflow_db = os.path.join(_project_root, 'mlruns', 'mlflow.db')
os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{mlflow_db}'

import mlflow
import wandb


def load_wandb_config():
    cfg_path = os.path.join(_project_root, '.wandb_config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


MIN_STEPS_UNNAMED = 10  # Discard unnamed runs with fewer steps


def deduplicate_runs(runs, client):
    """For duplicate run names, keep the run with the most metric data points."""
    name_groups = {}
    for r in runs:
        name = r.info.run_name or f'unnamed_{r.info.run_id[:8]}'
        if name not in name_groups:
            name_groups[name] = []
        name_groups[name].append(r)

    deduped = []
    skipped_dupes = 0
    for name, group in name_groups.items():
        if len(group) == 1:
            deduped.append((name, group[0]))
        else:
            # Pick the run with the most steps in any metric
            best = None
            best_steps = -1
            for r in group:
                try:
                    first_metric = list(r.data.metrics.keys())[0]
                    hist = client.get_metric_history(r.info.run_id, first_metric)
                    steps = len(hist)
                except Exception:
                    steps = 0
                if steps > best_steps:
                    best = r
                    best_steps = steps
            deduped.append((name, best))
            skipped_dupes += len(group) - 1

    return deduped, skipped_dupes


def should_skip(name, run, client):
    """Check if a run should be skipped."""
    # No metrics at all
    if not run.data.metrics:
        return True, "no metrics"

    # Unnamed runs with very few steps
    if name.startswith('None_') or name.startswith('unnamed_'):
        try:
            first_metric = list(run.data.metrics.keys())[0]
            hist = client.get_metric_history(run.info.run_id, first_metric)
            if len(hist) < MIN_STEPS_UNNAMED:
                return True, f"unnamed with only {len(hist)} steps"
        except Exception:
            return True, "unnamed, can't read history"

    return False, ""


def migrate_experiment(experiment_id, wandb_project, wandb_entity, dry_run=False):
    """Migrate all runs from an MLflow experiment to W&B."""
    client = mlflow.MlflowClient()

    experiment = client.get_experiment(experiment_id)
    exp_name = experiment.name
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name} (id={experiment_id})")
    print(f"{'='*60}")

    # Get all runs
    runs = client.search_runs(
        experiment_id, max_results=500, order_by=['start_time ASC'],
    )
    runs_with_metrics = [r for r in runs if r.data.metrics]
    print(f"  Total: {len(runs)}, with metrics: {len(runs_with_metrics)}")

    # Deduplicate
    deduped, skipped_dupes = deduplicate_runs(runs_with_metrics, client)
    print(f"  After dedup: {len(deduped)} (skipped {skipped_dupes} duplicates)")

    # Filter
    to_migrate = []
    for name, run in deduped:
        skip, reason = should_skip(name, run, client)
        if skip:
            print(f"    SKIP: {name} ({reason})")
        else:
            to_migrate.append((name, run))

    print(f"  Will migrate: {len(to_migrate)} runs")

    if dry_run:
        for name, r in to_migrate:
            n_metrics = len(r.data.metrics)
            print(f"    [DRY] {name:<45} {r.info.status:<10} metrics={n_metrics}")
        return len(to_migrate)

    migrated = 0
    for i, (name, run) in enumerate(to_migrate):
        print(f"\n  [{i+1}/{len(to_migrate)}] {name} ...", end=" ", flush=True)

        # Collect all metric histories
        available_metrics = sorted(run.data.metrics.keys())
        step_data = {}
        for metric_key in available_metrics:
            try:
                history = client.get_metric_history(run.info.run_id, metric_key)
                for h in history:
                    if h.step not in step_data:
                        step_data[h.step] = {}
                    step_data[h.step][metric_key] = h.value
            except Exception:
                pass

        total_points = sum(len(v) for v in step_data.values())

        # Collect params
        params = run.data.params or {}

        # Create W&B run
        wb_config = {
            'mlflow_run_id': run.info.run_id,
            'mlflow_experiment': exp_name,
            'mlflow_status': run.info.status,
            **params,
        }

        wb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=name,
            group=exp_name,
            config=wb_config,
            tags=['migrated-from-mlflow', exp_name],
            notes=f"Migrated from MLflow '{exp_name}'",
            reinit=True,
        )

        # Log metrics preserving step order
        for step in sorted(step_data.keys()):
            wandb.log(step_data[step], step=step)

        # Set summary
        for key, value in run.data.metrics.items():
            wb_run.summary[key] = value
        wb_run.summary['mlflow_status'] = run.info.status

        wandb.finish()
        migrated += 1
        print(f"OK ({total_points} pts)")

    return migrated


def main():
    parser = argparse.ArgumentParser(description='Migrate MLflow data to W&B')
    parser.add_argument('--experiment-id', default=None,
                        help='Specific MLflow experiment ID (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be migrated')
    parser.add_argument('--project', default=None,
                        help='Override W&B project name')
    args = parser.parse_args()

    if not os.path.exists(mlflow_db):
        print(f"ERROR: MLflow database not found at {mlflow_db}")
        sys.exit(1)

    wandb_cfg = load_wandb_config()
    entity = wandb_cfg.get('entity')

    # Use the specified project or map from config
    target_project = args.project
    project_map = wandb_cfg.get('default_projects', {})

    client = mlflow.MlflowClient()

    if args.experiment_id:
        experiment_ids = [args.experiment_id]
    else:
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments if e.name != 'Default']
        print(f"Found {len(experiment_ids)} experiments")

    total = 0
    for exp_id in experiment_ids:
        exp = client.get_experiment(exp_id)
        wb_project = target_project or project_map.get(exp.name, exp.name)
        n = migrate_experiment(exp_id, wb_project, entity, dry_run=args.dry_run)
        total += n

    print(f"\n{'='*60}")
    action = "Would migrate" if args.dry_run else "Migrated"
    print(f"  {action} {total} runs total")
    if not args.dry_run and entity:
        print(f"  View at: https://wandb.ai/{entity}")


if __name__ == '__main__':
    main()
