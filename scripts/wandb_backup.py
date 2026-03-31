#!/usr/bin/env python
"""Back up key project artifacts to W&B and exercise all available features.

Uploads:
1. Baseline checkpoint (fixv23_best.ckpt) as versioned Artifact
2. results.tsv as W&B Table (queryable, filterable in UI)
3. Experiment docs as Artifacts
4. Current config files as Artifacts
5. Best recent checkpoints as Artifacts

Also tests: Tables, Alerts, Summary, Tags, Notes.

Usage:
    python -m scripts.wandb_backup
    python -m scripts.wandb_backup --dry-run
    python -m scripts.wandb_backup --skip-checkpoints
"""

import argparse
import csv
import os
import sys
import yaml
from glob import glob

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# W&B setup
if not os.environ.get('WANDB_API_KEY'):
    _key_path = os.path.expanduser('~/.wandb_api_key')
    if os.path.exists(_key_path):
        with open(_key_path) as f:
            os.environ['WANDB_API_KEY'] = f.read().strip()

import wandb


def load_wandb_config():
    cfg_path = os.path.join(_project_root, '.wandb_config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def upload_results_table(run):
    """Upload results.tsv as a W&B Table — queryable and filterable in UI."""
    tsv_path = os.path.join(_project_root, 'results.tsv')
    if not os.path.exists(tsv_path):
        print("  SKIP: results.tsv not found")
        return

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    if not rows:
        print("  SKIP: results.tsv is empty")
        return

    columns = list(rows[0].keys())
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row.get(c, '') for c in columns])

    wandb.log({"experiments/results": table})
    print(f"  Table: {len(rows)} experiments logged to experiments/results")

    # Also upload raw file as artifact
    artifact = wandb.Artifact('results-tsv', type='dataset',
                              description='Experiment results (5-segment trend analysis)')
    artifact.add_file(tsv_path)
    wandb.log_artifact(artifact)
    print(f"  Artifact: results.tsv uploaded")


def upload_docs(run):
    """Upload project docs as a versioned artifact."""
    docs_dir = os.path.join(_project_root, 'docs')
    if not os.path.isdir(docs_dir):
        print("  SKIP: docs/ not found")
        return

    artifact = wandb.Artifact('project-docs', type='documentation',
                              description='Experiment logs, plans, and audit docs')
    doc_count = 0
    for f in os.listdir(docs_dir):
        fpath = os.path.join(docs_dir, f)
        if os.path.isfile(fpath):
            artifact.add_file(fpath)
            doc_count += 1

    # Also add program.md and CLAUDE.md
    for extra in ['program.md', 'CLAUDE.md']:
        epath = os.path.join(_project_root, extra)
        if os.path.exists(epath):
            artifact.add_file(epath)
            doc_count += 1

    wandb.log_artifact(artifact)
    print(f"  Artifact: {doc_count} docs uploaded")


def upload_configs(run):
    """Upload all config YAML files as a versioned artifact."""
    config_dir = os.path.join(_project_root, 'config')
    if not os.path.isdir(config_dir):
        print("  SKIP: config/ not found")
        return

    artifact = wandb.Artifact('project-configs', type='config',
                              description='Hydra YAML configs for all experiments')
    count = 0
    for f in os.listdir(config_dir):
        if f.endswith(('.yaml', '.yml', '.py')):
            artifact.add_file(os.path.join(config_dir, f))
            count += 1

    wandb.log_artifact(artifact)
    print(f"  Artifact: {count} config files uploaded")


def upload_baseline_checkpoint(run):
    """Upload the key baseline checkpoint."""
    ckpt_path = '/tmp/fixv23_best.ckpt'
    if not os.path.exists(ckpt_path):
        print("  SKIP: /tmp/fixv23_best.ckpt not found")
        return

    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    artifact = wandb.Artifact('baseline-fixv23', type='model',
                              description='PG-Occ baseline checkpoint (fixv23_best) — '
                                          'the reference for all autoresearch experiments',
                              metadata={
                                  'source': 'fixv23_acc_depth',
                                  'size_mb': round(size_mb, 1),
                                  'losses': 'depth_warping(5.0) + ov_mse(10.0) + ov_cos(1.0) + '
                                            'depth_foundation(1.0) + depth_gt(0.05)',
                              })
    artifact.add_file(ckpt_path, name='fixv23_best.ckpt')
    wandb.log_artifact(artifact)
    print(f"  Artifact: fixv23_best.ckpt ({size_mb:.0f}MB) uploaded")


def upload_best_checkpoints(run, max_count=10):
    """Upload the best/most recent checkpoints (not 'last.ckpt')."""
    work_dirs = os.path.join(_project_root, 'work_dirs')
    if not os.path.isdir(work_dirs):
        print("  SKIP: work_dirs/ not found")
        return

    # Find all non-last checkpoints, sorted by modification time (newest first)
    ckpts = []
    for ckpt in glob(os.path.join(work_dirs, '**', '*.ckpt'), recursive=True):
        if 'last.ckpt' in ckpt:
            continue
        ckpts.append((os.path.getmtime(ckpt), ckpt))

    ckpts.sort(reverse=True)
    ckpts = ckpts[:max_count]

    if not ckpts:
        print("  SKIP: no checkpoints found")
        return

    artifact = wandb.Artifact('recent-best-checkpoints', type='model',
                              description=f'Top {len(ckpts)} most recent best checkpoints',
                              metadata={'count': len(ckpts)})

    total_size = 0
    for _, ckpt in ckpts:
        name = os.path.basename(os.path.dirname(ckpt)) + '/' + os.path.basename(ckpt)
        artifact.add_file(ckpt, name=name)
        total_size += os.path.getsize(ckpt)

    wandb.log_artifact(artifact)
    print(f"  Artifact: {len(ckpts)} checkpoints ({total_size / (1024**3):.1f}GB) uploaded")
    for _, ckpt in ckpts:
        run_dir = os.path.basename(os.path.dirname(ckpt))
        print(f"    - {run_dir}/{os.path.basename(ckpt)}")


def upload_code_snapshot(run):
    """Upload key source files as a code artifact for reproducibility."""
    artifact = wandb.Artifact('source-code', type='code',
                              description='Key model and training source files')
    code_files = [
        'models/pgocc/pgocc.py',
        'models/pgocc/render.py',
        'models/pgocc/loss_utils.py',
        'models/pgocc/sparse_gaussians_decoder.py',
        'models/pgocc/gaussian_prediction.py',
        'models/gausstr.py',
        'models/vitdet_fpn.py',
        'models/gausstr_decoder.py',
        'models/gausstr_head.py',
        'scripts/train.py',
        'scripts/train_pgocc.py',
        'scripts/eval_experiment.py',
        'dataset/dataset.py',
        'dataset/datamodule.py',
    ]
    count = 0
    for f in code_files:
        fpath = os.path.join(_project_root, f)
        if os.path.exists(fpath):
            artifact.add_file(fpath, name=f)
            count += 1

    wandb.log_artifact(artifact)
    print(f"  Artifact: {count} source files uploaded")


def create_summary_metrics(run):
    """Log summary metrics from results.tsv as a W&B bar chart."""
    tsv_path = os.path.join(_project_root, 'results.tsv')
    if not os.path.exists(tsv_path):
        return

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    keep_count = sum(1 for r in rows if r.get('status') == 'keep')
    discard_count = sum(1 for r in rows if r.get('status') == 'discard')
    crash_count = sum(1 for r in rows if r.get('status') in ('crash', 'timeout'))
    total = len(rows)

    run.summary['total_experiments'] = total
    run.summary['experiments_kept'] = keep_count
    run.summary['experiments_discarded'] = discard_count
    run.summary['experiments_crashed'] = crash_count
    run.summary['keep_rate'] = keep_count / total if total > 0 else 0

    # Summary table
    summary_table = wandb.Table(
        columns=['category', 'count', 'percentage'],
        data=[
            ['kept', keep_count, f'{keep_count/total*100:.1f}%' if total else '0%'],
            ['discarded', discard_count, f'{discard_count/total*100:.1f}%' if total else '0%'],
            ['crashed', crash_count, f'{crash_count/total*100:.1f}%' if total else '0%'],
        ]
    )
    wandb.log({"experiments/summary": summary_table})
    print(f"  Summary: {total} experiments ({keep_count} kept, {discard_count} discarded, {crash_count} crashed)")


def main():
    parser = argparse.ArgumentParser(description='Back up project artifacts to W&B')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-checkpoints', action='store_true',
                        help='Skip uploading large checkpoint files')
    parser.add_argument('--max-checkpoints', type=int, default=10,
                        help='Max number of recent checkpoints to upload')
    args = parser.parse_args()

    wandb_cfg = load_wandb_config()
    entity = wandb_cfg.get('entity')
    project = wandb_cfg.get('default_projects', {}).get('pgocc_t4', 'New Perception Dev')

    if args.dry_run:
        print(f"DRY RUN — would upload to {entity}/{project}")
        print(f"  Baseline ckpt: {os.path.exists('/tmp/fixv23_best.ckpt')}")
        print(f"  Results TSV: {os.path.exists(os.path.join(_project_root, 'results.tsv'))}")
        print(f"  Docs: {os.path.isdir(os.path.join(_project_root, 'docs'))}")
        print(f"  Configs: {os.path.isdir(os.path.join(_project_root, 'config'))}")
        return

    print(f"Backing up to: {entity}/{project}")
    print("=" * 60)

    run = wandb.init(
        project=project,
        entity=entity,
        name='project-backup',
        job_type='backup',
        tags=['backup', 'full-project'],
        notes='Full project backup: configs, docs, code, checkpoints, experiment results',
    )

    # 1. Results table (W&B Tables — queryable in UI)
    print("\n[1/7] Results table...")
    upload_results_table(run)

    # 2. Experiment summary metrics
    print("\n[2/7] Summary metrics...")
    create_summary_metrics(run)

    # 3. Documentation
    print("\n[3/7] Project docs...")
    upload_docs(run)

    # 4. Config files
    print("\n[4/7] Config files...")
    upload_configs(run)

    # 5. Source code snapshot
    print("\n[5/7] Source code snapshot...")
    upload_code_snapshot(run)

    # 6. Baseline checkpoint
    print("\n[6/7] Baseline checkpoint...")
    if not args.skip_checkpoints:
        upload_baseline_checkpoint(run)
    else:
        print("  SKIP (--skip-checkpoints)")

    # 7. Recent best checkpoints
    print("\n[7/7] Recent best checkpoints...")
    if not args.skip_checkpoints:
        upload_best_checkpoints(run, max_count=args.max_checkpoints)
    else:
        print("  SKIP (--skip-checkpoints)")

    # Send alert
    wandb.alert(
        title="Project Backup Complete",
        text=f"Full backup uploaded to {entity}/{project}. "
             f"Includes: results table, docs, configs, code, checkpoints.",
        level=wandb.AlertLevel.INFO,
    )

    print("\n" + "=" * 60)
    print(f"Backup complete! View at: {run.url}")
    wandb.finish()


if __name__ == '__main__':
    main()
