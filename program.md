# PG-Occ Autoresearch Program

This is the autonomous research program for PG-Occ (Progressive Gaussian Occupancy). You are a fully autonomous ML researcher. Your job is to improve the model's self-supervised 3D occupancy prediction quality through systematic experimentation.

## Setup

To set up a new experiment session:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read context files** (you MUST read these for full context, use Read tool on each):
   - `program.md` — this file (your operating instructions)
   - `docs/PGOCC_EXPERIMENT_LOG.md` — history of all experiments and lessons learned
   - `docs/PGOCC_SELFOCCFLOW_INTEGRATION_PLAN.md` — feature roadmap with SelfOccFlow phases
   - `config/pgocc_t4.yaml` — current training config
   - `models/pgocc/pgocc.py` — main model (you modify this)
   - `models/pgocc/render.py` — rendering code (you modify this)
   - `models/pgocc/loss_utils.py` — loss functions (you modify this)
   - `models/pgocc/sparse_gaussians_decoder.py` — decoder (you modify this)
4. **Verify baseline checkpoint exists**: `ls -la /tmp/fixv23_best.ckpt`
5. **Check GPU availability**: `nvidia-smi` — need at least 4 free GPUs.
6. **Initialize results.tsv**: Create if not exists (eval script handles this automatically).
7. **Generate first experiment name**: `python -m scripts.eval_experiment --next-name` → e.g. `ar_mar12_001`
8. **Confirm and go**: Confirm setup looks good, then start the loop.

## Experiment Naming

All experiments use the auto-naming system: `ar_<month><day>_<NNN>`

```bash
# Generate next name automatically (increments from last used number today)
python -m scripts.eval_experiment --next-name
# → ar_mar12_001, ar_mar12_002, ar_mar12_003, ...

# List all experiments from a session
python -m scripts.eval_experiment --list ar_mar12
```

The naming is automatic — you never hardcode names. The eval script tracks used numbers via both W&B and results.tsv.

## Experimentation

Each experiment runs on 4 GPUs for a **fixed budget of 3000 steps** (~45 minutes). This is enough to evaluate whether a change improves convergence trends. You launch experiments as:

```bash
source .venv/bin/activate && \
timeout 4500 python -m scripts.train_pgocc \
  --config-name pgocc_t4 \
  run_name=<experiment_name> \
  +load_from=/tmp/fixv23_best.ckpt \
  trainer.devices=4 \
  +trainer.max_steps=3000 \
  trainer.max_epochs=1 \
  data.num_workers=4 \
  <additional overrides> \
  > /tmp/<experiment_name>.log 2>&1
```

NOTE: Use `trainer.max_epochs=1` NOT `max_epochs=999`. With max_epochs=1, `estimated_stepping_batches=max_steps`, and `steps_per_epoch=max_steps//1=max_steps`, which correctly activates warmup fractions. With max_epochs=999, steps_per_epoch≈1 and all warmup fractions are disabled.


### What You CAN Modify

- `models/pgocc/pgocc.py` — training losses, masks, rendering strategy, loss weights
- `models/pgocc/render.py` — depth loss formulation, rendering helpers
- `models/pgocc/loss_utils.py` — warp loss, new loss functions
- `models/pgocc/sparse_gaussians_decoder.py` — decoder architecture, temporal processing
- `models/pgocc/gaussian_prediction.py` — Gaussian prediction dataclass
- `config/pgocc_t4.yaml` — training hyperparameters, loss weights

### What You CANNOT Modify

- `scripts/train_pgocc.py` — training entrypoint (fixed infrastructure)
- `scripts/eval_experiment.py` — evaluation harness (fixed metric)
- `dataset/` — data pipeline (read-only)
- `models/pgocc/utils.py` — utility functions (read-only)
- Package dependencies

### The Goal

**Improve self-supervised 3D occupancy quality.** The primary metrics (all lower = better):

| Metric | What it measures | Priority |
|--------|-----------------|----------|
| `train/depth_0` | Foundation depth alignment | HIGH — geometry quality |
| `train/ov_cos_0` | Open-vocabulary feature quality | HIGH — semantic quality |
| `train/warp_0` | Temporal depth consistency | HIGH — self-supervised signal |
| `train/depth_gt_0` | Sparse LiDAR depth accuracy | MEDIUM — ground truth reference |
| `train/sem_ce_0` | Semantic classification | LOW — auxiliary task |

The **compound objective** is: improve depth_0 and ov_cos_0 trends without regressing warp. A change that improves warp but destroys ov_cos is not useful. A change that mildly improves everything is ideal.

### Simplicity Criterion

All else being equal, simpler is better. Removing code that doesn't help is a win. Adding 50 lines for 1% improvement is questionable. Adding 10 lines for 5% improvement is great.

## Evaluation Protocol — MANDATORY

**NEVER cherry-pick individual data points.** Self-supervised training is noisy. You MUST use sliding-average trend analysis.

After each experiment completes, evaluate with:

```bash
python -m scripts.eval_experiment --run-name <experiment_name>
```

This computes **5-segment sliding averages** and outputs IMPROVED/REGRESSED/FLAT for each metric.

If the eval script isn't available or you need manual analysis, follow this protocol exactly:

1. Divide all logged data points into **5 equal segments**
2. Compute the **average** of each segment
3. Compare Seg5 average to Seg1 average as **percentage change**
4. Verdict: `< -2%` = IMPROVED, `> +5%` = REGRESSED, else FLAT
5. NEVER report "step X had value Y therefore it's improving" — that's cherry-picking

## Output Format

The training script logs to W&B. After completion, extract key metrics:

```bash
python -m scripts.eval_experiment --run-name <experiment_name> --json
```

## Logging Results

Log every experiment to `results.tsv` (tab-separated):

```
commit	experiment	depth_0_pct	ov_cos_pct	warp_pct	depth_gt_pct	status	description
```

- commit: git short hash (7 chars)
- experiment: run name
- *_pct: percentage change from 5-segment analysis (negative = improved)
- status: `keep`, `discard`, or `crash`
- description: what this experiment tried

Example:
```
commit	experiment	depth_0_pct	ov_cos_pct	warp_pct	depth_gt_pct	status	description
a1b2c3d	baseline_3k	0.0	0.0	0.0	0.0	keep	baseline 3000-step run
b2c3d4e	lower_warp_w	-3.2	+1.1	-5.4	-2.1	keep	depth_warping 3.0 → 2.0
c3d4e5f	add_entropy	+2.5	+8.3	+0.1	+3.2	discard	alpha entropy regularization
```

## The Experiment Loop

Each iteration runs inline — no mode switching, no pausing. Analyze, code, run, evaluate, repeat.

```
LOOP FOREVER:
  1. Analyze → 2. Code → 3. Name → 4. Run (foreground) → 5. Evaluate → 6. Decide → repeat
```

1. **Analyze**: Read `results.tsv` and recent `docs/PGOCC_EXPERIMENT_LOG.md`.
   - Last experiment KEEP → it's the new baseline, build on it
   - Last experiment DISCARD → understand why, try a different direction
   - Out of ideas → WebSearch for new papers, re-read `PGOCC_SELFOCCFLOW_INTEGRATION_PLAN.md`

2. **Code** (if needed): Apply changes. Small diffs only. Commit:
   ```bash
   git commit -m "exp: <description>"
   ```
   (If config-only change, commit the YAML. If no code change needed, skip commit.)

3. **Name**: Generate experiment name:
   ```bash
   EXP_NAME=$(python -m scripts.eval_experiment --next-name)
   ```

4. **Run** (foreground, blocking, with timeout):
   ```bash
   source .venv/bin/activate && \
   timeout 3600 python -m scripts.train_pgocc \
     --config-name pgocc_t4 \
     run_name=$EXP_NAME \
     +load_from=/tmp/fixv23_best.ckpt \
     trainer.devices=4 \
     +trainer.limit_train_batches=3000 \
     trainer.max_epochs=1 \
     data.num_workers=4 \
     <overrides> \
     > /tmp/$EXP_NAME.log 2>&1
   ```
   **Do NOT use `nohup ... &` or `run_in_background`.** Run foreground and wait for it to finish.
   Do NOT use tee. Do NOT let output flood context.

5. **Evaluate**:
   ```bash
   python -m scripts.eval_experiment --run-name $EXP_NAME --log --description "<hypothesis>"
   ```

6. **Decide**:
   - Exit code 0 (KEEP) → keep the commit, it's the new baseline
   - Exit code 1 (DISCARD) → `git reset --hard HEAD~1` to revert

7. **Update log**: After every 3-5 experiments, append summary to `docs/PGOCC_EXPERIMENT_LOG.md`.

8. **Go to step 1. NEVER STOP.**

## Crash Recovery

If training crashes (non-zero exit code, NOT 124):

1. Read the last 50 lines of the log:
   ```bash
   tail -n 50 /tmp/$EXP_NAME.log
   ```
2. **Obvious fix** (import error, typo, shape mismatch, CUDA OOM):
   - Fix the code
   - `git commit --amend --no-edit` (same commit)
   - Re-run with **same** `$EXP_NAME`
3. **Unfixable** (unclear error, repeated crash):
   - Log `crash` status in results.tsv manually
   - `git reset --hard HEAD~1` to revert
   - Move on to next experiment
4. **Max 1 retry per crash.** Never get stuck on a crash.

## Timeout Protection

If `timeout 3600` kills the process (exit code 124):

1. Log `timeout` status manually in results.tsv
2. `git reset --hard HEAD~1`
3. Continue to next experiment
4. If 2+ consecutive timeouts → reduce `limit_train_batches` to 2000

## Experiment Ideas — NOT Limited to These

The ideas below are a starting point, NOT a ceiling. You are strongly encouraged to:

1. **Search for new papers**: Use WebSearch to find recent self-supervised occupancy / NeRF / Gaussian splatting papers. Look for novel loss functions, training tricks, and architectural ideas.
2. **Introduce entirely new losses**: If a paper proposes a loss that fits our pipeline, implement it. You are not limited to the existing loss suite.
3. **Borrow from adjacent fields**: Depth estimation, optical flow, scene flow, novel view synthesis — all have transferable ideas.
4. **Read source code**: If a paper has a GitHub repo, read their loss implementations for details the paper glosses over.

### Research Strategy (inline — no mode switching)

When you run out of ideas from the queue below, do active research inline:

1. WebSearch "self-supervised 3D occupancy prediction loss 2024 2025"
2. WebSearch "gaussian splatting occupancy training tricks"
3. WebSearch "CF-OccFlow coarse-to-fine occupancy flow"
4. Read promising paper abstracts/methods sections
5. Identify 2-3 transplantable ideas → pick one → proceed to step 2 of the loop

### Key Papers to Reference

These papers have ideas directly applicable to our pipeline:

- **SelfOccFlow** — static/dynamic disentanglement, temporal aggregation, dynamic coverage loss (integration plan: `docs/PGOCC_SELFOCCFLOW_INTEGRATION_PLAN.md`)
- **CF-OccFlow** — coarse-to-fine occupancy flow, multi-scale consistency, flow-based temporal supervision
- **GaussTR** (our own codebase) — `loss_text` (text contrastive, weight 3.0), `loss_position` (direct xyz supervision), `loss_ce` (semantic CE)
- **PG-Occ** (original paper) — progressive Gaussian decoding, multi-stage rendering
- **OccNeRF / RenderOcc / SurroundOcc** — various self-supervised occupancy approaches with different loss formulations
- **MonoDepth2** — auto-masking, minimum reprojection loss (we already use this in warp)
- **GaussianFormer / GaussianOcc** — Gaussian-based occupancy with different training strategies

### Starting Ideas (expand freely)

#### Tier 1: Tuning Existing Losses
1. Loss weight ratios (depth_warping, depth_foundation, ov_cos)
2. Alpha threshold (0.05 / 0.1 / 0.2)
3. Warp warmup schedule (0.1 / 0.25 / 0.5 epochs)
4. Depth loss formulation (SiLog vs L1 ratio)
5. Learning rate / schedule tuning

#### Tier 2: SelfOccFlow-Inspired (PRIORITY — previously blocked by alpha_mask bug, need clean re-test)

**Phase 1 (code exists in pgocc.py, needs clean ablation):**
6. Dynamic coverage loss (dyn_cov) — BCE(alpha, 1.0) on dynamic SAM3 pixels. ALREADY CODED but never tested in isolation on clean baseline.
7. Dynamic depth loss (dyn_depth) — L1(depth, foundation_depth) on dynamic pixels. ALREADY CODED.

**Phase 2 (code exists, never tested on clean baseline):**
8. Static-only warp rendering — render depth from `opacity * p_static.detach()`, use for warp. Code in pgocc.py lines 572-591. Activates when branch_cls > 0.
9. Branch classification (branch_cls) — BCE on branch_head logits vs SAM3 projected labels. Code in decoder. All previous tests (fixv31-34) were contaminated by alpha_mask bug.

**Phase 3 (NOT YET IMPLEMENTED — highest expected ROI):**
10. **Static temporal aggregation** — Transform neighbor frames' static Gaussians to current ego frame, concatenate with current static Gaussians, render denser static scene. This is SelfOccFlow's most valuable idea. Requires: (a) neighbor frame Gaussian predictions, (b) ego transforms between frames (already in data), (c) static branch probs (from Phase 2). Implementation goes in pgocc.py training_step.
11. **Bidirectional temporal data** — Add future sweeps to annotation prep (Phase 0 infra). Currently only past frames available.

**Phase 4 (NOT YET IMPLEMENTED):**
12. Motion head — per-dynamic-query XY offsets for temporal alignment
13. Similarity-flow pseudo-labels — BEV feature matching between adjacent frames

#### Tier 3: New Losses to Explore
10. **Depth smoothness / edge-aware loss**: Penalize depth discontinuities except at image edges
11. **Multi-scale reprojection**: Warp loss at 1x, 0.5x, 0.25x resolution
12. **Feature consistency loss**: Rendered OV features should be consistent across adjacent frames for static regions
13. **Gaussian position loss**: Supervise xyz with unprojected depth (GaussTR's `loss_position`)
14. **Scale regularization**: Penalize extreme Gaussian scales
15. **Opacity sharpness**: Encourage binary opacity (0 or 1) via entropy loss
16. **Normal consistency**: Derive surface normals from depth, enforce smoothness
17. **Occupancy flow loss**: From CF-OccFlow — temporal occupancy consistency via flow warping
18. **Contrastive depth**: Positive pairs = same-object pixels, negative = different depths
19. **Rendering quality loss**: LPIPS or other perceptual losses on rendered vs GT images

#### Tier 4: Architectural Changes
20. All-stage inference (merge 3 progressive stages)
21. Query count / decoder depth tuning
22. Gradient clipping tuning
23. Temporal attention mechanisms (with direct supervision this time)
24. Auxiliary BEV head for spatial reasoning
25. Multi-frame feature aggregation before decoder

## Known Constraints and Pitfalls

Read these carefully. Every item was learned from a failed experiment.

1. **alpha_mask on warp = DEATH**: Do NOT add `alpha_mask` to the warp pixel mask. fixv23 proved this kills convergence. `warp_full_mask = warp_pixel_mask` ONLY.

2. **alpha_mask on depth IS REQUIRED**: alpha=0 → depth_ed=0.1 (clamp) → SiLog loss vs 20m target explodes. Keep alpha_mask(>0.1) on depth_foundation and depth_gt.

3. **sem_ce/sem_text need .detach() on OV features**: Without detach, classification gradients corrupt OV features, causing ov_cos to spike +79%. With detach: only +8%.

4. **Batch size > 1 crashes**: gsplat uses camera dim as batch. Use `accumulate_grad_batches` instead.

5. **Hard temporal mask needs supervision**: Random branch_probs ≈ 0.5 corrupt past-frame features. Only enable hard mask when `branch_cls > 0`.

6. **Temporal gate is gradient-dead**: Gradient path loss→render→voxelize→gaussian→decoder→gate is too long. Gate entropy stays at max after 1000 steps.

7. **Warp warmup causes V-shape**: depth/OV losses rise then recover as warp_factor ramps 0→1. This is expected, not a regression.

8. **T4 is 10Hz with 35% stationary samples**: min_translation=0.5m filters these from warp loss automatically.

9. **DDP requires all ranks to call same allreduces**: Always log diagnostic dict keys even for stationary samples (fill with 0.0).

## Important Files Reference

```
models/pgocc/pgocc.py              — Main model, training_step(), all losses
models/pgocc/render.py             — batch_splatting_render(), get_depth_loss()
models/pgocc/loss_utils.py         — calc_time_warping_loss(), SSIM, backprojection
models/pgocc/sparse_gaussians_decoder.py  — Progressive decoder, branch_heads
models/pgocc/gaussian_prediction.py       — GaussianPrediction dataclass
config/pgocc_t4.yaml               — All hyperparameters
docs/PGOCC_EXPERIMENT_LOG.md       — Historical experiment log (ALWAYS UPDATE)
docs/PGOCC_SELFOCCFLOW_INTEGRATION_PLAN.md — Feature roadmap
```

## SelfOccFlow Integration Status

The integration plan is at `docs/PGOCC_SELFOCCFLOW_INTEGRATION_PLAN.md`. Current status:

| Phase | Content | Status |
|-------|---------|--------|
| Phase 0 | Bidirectional temporal data (future sweeps) | NOT DONE |
| Phase 1 | dyn_cov + dyn_depth losses | CODE EXISTS — never cleanly ablated (alpha_mask bug contaminated all prior runs) |
| Phase 2 | branch_head + static-only warp rendering | CODE EXISTS — branch_head in decoder, static warp in pgocc.py. Never tested on clean baseline |
| Phase 3 | Static temporal aggregation | NOT IMPLEMENTED — highest expected value |
| Phase 4 | Motion head + similarity flow | NOT IMPLEMENTED |

**All fixv31-38 experiments were contaminated by the alpha_mask-on-warp bug.** fixv39 was the first clean run. SelfOccFlow phases need re-testing from scratch on the clean baseline.

## Baseline Reference

- **Checkpoint**: `/tmp/fixv23_best.ckpt`
- **Architecture**: ResNet50+FPN → SparseGaussiansDecoder (3-stage progressive)
- **Losses**: depth_warping(5.0) + ov_mse(10.0) + ov_cos(1.0) + depth_foundation(1.0) + depth_gt(0.05)
- **Known behavior**: depth_0 improves, ov_cos improves, warp is flat at ~0.15 after warmup
- **This checkpoint is READ-ONLY**: Always load with `+load_from=/tmp/fixv23_best.ckpt`

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should
continue. Do NOT ask "should I keep going?" or "is this a good stopping point?"
The human might be asleep and expects you to work **indefinitely** until manually
stopped.

Do NOT use background tasks or polling loops. Run each experiment as a foreground
blocking command. When it finishes, evaluate immediately and start the next one.

If you run out of ideas: re-read the experiment log, SelfOccFlow plan, search for
new papers, try combining near-misses, try the opposite of what failed.

As a reference: each experiment takes ~45 minutes. You can run ~30 experiments per
day. The human wakes up to a results.tsv with 10+ experiments all completed by you.
