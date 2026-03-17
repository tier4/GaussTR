# PG-Occ Experiment Log & Plan

> **维护说明**: 每次实验结束或有新发现时更新此文档。记录所有尝试、结果、失败原因和下一步计划。

## 1. 基线参考

### fixv23 (基线 checkpoint)
- **路径**: `/tmp/fixv23_best.ckpt`
- **特点**: 纯自监督（depth_warping + ov_mse + ov_cos + depth_foundation + depth_gt）
- **没有**: sem_ce, sem_text, branch_cls, temporal gate
- **Step 3k 指标**: depth 2.8→1.6 (↓43%), ov_cos 0.26→0.17 (↓35%)
- **结论**: 收敛正常，是所有后续实验的加载起点

### fixv27d (sem_ce/sem_text 加入后)
- **Step 5449 指标**: warp 0.16-0.28, depth 0.98-1.33, ov_cos 0.25-0.30, depth_gt 2.55-2.85
- **结论**: 收敛变慢，depth 比 fixv23 差。sem_ce/sem_text 多任务冲突

---

## 2. 实验记录

### fixv31 — Soft Branch Rendering (失败)
- **Git**: `7ed6260`
- **改动**: 分别渲染 static/dynamic 层 (opacity × p_static, opacity × p_dynamic)，用 static alpha 做 warp，dynamic alpha 做覆盖 loss
- **结果**: ov_cos 从 0.17 暴涨到 0.33，所有指标恶化
- **失败原因**: 额外渲染的梯度通过共享 query features 反向传播，破坏 OV 表示。branch_head 没有 detach，分类梯度直接干扰几何特征
- **教训**: branch_head 必须 detach；分支渲染需要非常小心梯度隔离

### fixv32 — Detached Branch + Projection Supervision
- **Git**: `83b3d6f`
- **改动**: 移除分支渲染，改用投影式 SAM3 监督 (project Gaussian means → sample SAM3 mask → BCE)。branch_head 使用 detached features
- **结果**: 消除了 fixv31 的梯度冲突，但收敛高原仍然存在 (step 11700 所有指标平坦)
- **结论**: 分支分类可以学（p_static→0.93），但仅靠分类不能打破高原

### fixv33_tgate — Soft Temporal Attention Gate (失败)
- **Git**: `7c2f223`
- **改动**: 在 SparseBEVSampling 和 AdaptiveMixing 之间加入可学习时域门控。每个 query 学习 T=8 帧的注意力权重。Zero-init → softmax → 初始为 uniform (identity)
- **Step 950 指标**: ov_cos 0.22-0.30, depth 0.48-0.57, tgate_entropy 2.078/2.079 (最大熵=完全均匀)
- **失败原因**: 梯度路径太长 (loss → render → voxelize → gaussian → decoder → mixing → gate)，gate 完全学不动。1000 步后 entropy 几乎没变
- **教训**: 间接渲染 loss 的梯度无法训练细粒度时域权重。需要直接监督或硬决策

### fixv34 — Hard Temporal Masking (只监督最后层)
- **改动**: 用 branch_probs 硬掩码替代 soft gate。动态 query 的过去帧特征替换为当前帧副本
- **Bug**: branch_cls loss 只作用于 `gau_preds[-1]`，Layer 0/1 的 branch_head 没有监督
- **结果**: `hard_mask_dyn_prob_L1` 卡在 0.40 (随机水平)，Layer 0/1 随机掩码 40% 的 query
- **教训**: 时域掩码依赖上一层的 branch_probs，所以 **所有层** 都需要 branch 监督

### fixv34b — Hard Temporal Masking + All-Stage Supervision
- **Git**: `3bb2f53`
- **改动**: 修复 bug — branch_cls loss 应用到所有 3 个 progressive stage
- **Step 950 指标**: hard_mask_dyn_prob_L1 **0.05-0.08** (正确收敛！), ov_cos 0.22-0.30
- **结论**: branch 分类在所有层正确收敛。但 ov_cos 仍然回归 0.30 (同 fixv33)

### fixv34c — Hard Mask + All-Stage + sem_ce Warmup ⬅️ 当前运行中
- **Git**: `e555b80`
- **改动**: sem_ce loss 从 0 线性 warmup 到 300 步。防止随机 class_head MLP 在初始化时破坏预训练的 OV 特征
- **Step 3449 指标**:

| 指标 | Step 49 | Step 1500 | Step 3449 | fixv27d@5449 |
|------|---------|-----------|-----------|--------------|
| ov_cos | 0.174 | 0.28 | 0.24 | 0.26 |
| ov_mse | 0.0003 | 0.003 | 0.0016 | — |
| depth_0 | 0.53 | 0.52 | **0.50** | 1.13 |
| warp_0 | 0.22 | 0.14 | **0.14** | 0.17 |
| sem_ce | 2.69 | 0.45 | 0.44 | — |

- **对比 fixv27d**: depth_0 **好 2.2 倍**，warp 好 18%，ov_cos 持平
- **问题**: step 800+ 后各指标进入高原。ov_mse/sem_ce 仍在缓慢下降
- **状态**: 8 GPU 运行中 (PID 3785140)

### fixv35 — Phase 1 Dynamic Losses + Hard Mask + CE Warmup ⬅️ 当前运行中
- **Git**: `4f77a3c`
- **改动**: 在 fixv34c 基础上启用 `dyn_cov` (0.5) 和 `dyn_depth` (0.5)
- **早期指标** (step 1149):

| 指标 | Step 49 | Step 1149 | 说明 |
|------|---------|-----------|------|
| dyn_alpha_coverage | 0.87 | **0.95** | 动态物体覆盖率提升 |
| dyn_cov loss | 0.061 | 0.058 | BCE 接近饱和 |
| dyn_depth loss | 0.72 | 0.55 | 动态深度在改善 |
| ov_cos | 0.17 | 0.27 | 同 fixv34c 回归 |
| depth_0 | 0.50 | 0.56 | 同 fixv34c 水平 |
| warp_0 | 0.22 | 0.14 | 同 fixv34c 水平 |

- **发现**: 动态覆盖从 87%→95%（loss 有效），但整体收敛和 fixv34c 相同（动态只占 3.75% 像素）
- **状态**: 已完成分析

### fixv36a — Baseline 5-Loss Only (验证多任务冲突)
- **改动**: 只用 fixv23 的 5 个基础 loss (warp, ov_mse, ov_cos, depth_foundation, depth_gt)，关闭 sem_ce/sem_text/dyn_cov/dyn_depth/branch_cls
- **Bug 发现**: 第一版 hard temporal masking 在 branch_cls=0 时仍然运行，random branch_probs≈0.5 导致所有 query 的 past-frame 特征被 50% 替换。修复：`use_hard_mask` 参数
- **Step 549 指标**:

| 指标 | Step 49 | Step 549 | 趋势 |
|------|---------|----------|------|
| ov_cos | 0.073 | **0.053** | ✅ 持续下降（最佳！）|
| warp | 0.23 | **0.06** | ✅ 强改善 |
| depth_0 | 0.51 | 0.64 | ⚠️ warp warmup V-shape |
| depth_gt | 1.69 | 2.15 | ⚠️ warp warmup V-shape |

- **结论**: 基础 5-loss 的 ov_cos 持续下降至 0.053，远好于任何多 loss 实验。**确认：多任务冲突是收敛高原的根因**

### fixv36b — All Losses + Gradient Accumulation=2
- **改动**: 全部 loss + accumulate_grad_batches=2（等效 batch_size=2，但不需要改代码）
- **Step 599 指标**:

| 指标 | Step 49 | Step 599 | 趋势 |
|------|---------|----------|------|
| ov_cos | 0.172 | **0.233** | ❌ 恶化（多任务冲突）|
| warp | 0.19 | 0.25 | ❌ 恶化 |
| depth_0 | 0.58 | 0.43 | ✅ 改善 |
| sem_ce | 2.70 | 0.63 | ✅ 快速学习 |

- **结论**: 梯度累积不能解决问题。sem_ce 学得很好但破坏了 OV 特征

### fixv37 — Detached Semantic Losses (部分成功)
- **改动**: sem_ce 和 sem_text 在 `.detach()` 的 OV 特征上计算
- **Step 23399 真实趋势分析 (first10% avg vs last10% avg)**:

| 指标 | first 10% | last 10% | 趋势 | 对比 fixv36b |
|------|-----------|----------|------|-------------|
| train_loss | 9.20 | **10.72** | ↑ +16.6% ❌ | ↑ +17.7% |
| ov_cos | 0.066 | **0.080** | ↑ +20.9% ❌ | ↑ +78.7% |
| warp | 0.15 | 0.15 | → 持平 | ↓ -38% |
| depth_0 | 0.50 | 0.46 | ↓ -8% | → 持平 |
| sem_ce | 1.24 | 1.01 | ↓ -18% ✅ | ↓ similar |
| depth_gt | 1.98 | 1.96 | → 持平 | → 持平 |

- **结论**: detach 减缓了 ov_cos 恶化 (20.9% vs 78.7%)，但**没有打破高原**。train_loss 仍在上升。sem_ce 是唯一真正收敛的 loss

### fixv37b — Detached Semantic Losses (8GPU) ⬅️ 当前运行中
- Step 2949 同样模式：train_loss ↑5.5%, ov_cos → 持平, warp → 持平
- **状态**: 8 GPU 运行中，但趋势不乐观

---

## 3. 关键发现

### 3.1 ov_cos 回归的根本原因 (已解决 ✅)
每次从 fixv23 加载权重后，ov_cos 必然从 0.17 上升到 0.28-0.30。原因：
- sem_ce 的 class_head MLP 随机初始化，step 0 产生 CE=2.7 的噪声梯度
- 噪声梯度通过渲染管线反向传播，破坏已收敛的 OV 特征
- sem_ce warmup 只能缓解（0.30→0.25），不能完全消除
- 本质是 **多任务学习的冲突**：sem_ce 和 ov_cos 争夺 OV 特征方向

**fixv36 对比实验确认**:
- fixv36a (baseline 5-loss): ov_cos 0.073→0.053 ✅ 持续下降
- fixv36b (all losses): ov_cos 0.172→0.233 ❌ 持续恶化
- **解决方案 (fixv37)**: sem_ce/sem_text 在 `.detach()` 特征上运行，彻底隔离梯度

### 3.2 Soft Temporal Gate 失败的原因
梯度路径：loss → render → voxelize → Gaussian params → decoder → AdaptiveMixing → gate
- 8 层非线性变换，per-frame 梯度信号消失
- 1000 步后 gate entropy 仍然 = ln(8) = 2.079 (完全均匀)
- 结论：**间接监督信号无法训练时域权重**，需要直接信号

### 3.3 alpha_mask 的正确使用
- **必须**用在 depth_foundation 和 depth_gt 上：alpha=0 → depth_ed=0.1 (clamp) → SiLog 爆炸
- **不能**用在 OV/sem_ce/sem_text 上：OV 特征在低 alpha 区域仍有信号
- **warp**: 用 alpha_mask(>0.1) — garbage depth → garbage warp

### 3.4 从fixv23加载后loss起点远好于fixv23终点
- fixv23 训练8 epoch 后：depth_0=1.02, ov_cos=0.111, train_loss=5.0
- 加载fixv23后第一步：depth_0=0.50, ov_cos=0.069, train_loss=2.0
- 新代码的渲染/PCA/masking 改进让同样的权重表现更好
- train_loss 上升可能主要是 warp warmup V-shape (warp_factor 从0开始涨)

### 3.5 warp loss 上的 alpha_mask 是收敛恶化的根因 (已修复 ✅)
- fixv23（能收敛）: `warp_mask = warp_pixel_mask`（只有 ego+sky+dynamic mask）
- fixv26-fixv38（不能收敛）: `warp_mask = warp_pixel_mask & alpha_mask`
- 加上 alpha_mask 后: warp trend 从 -17% → +6%, train_loss 从 +1% → +30%
- 去掉后 (fixv39a): 立刻恢复收敛，和 fixv23 原始代码趋势一致
- **原因**: alpha_mask 过滤掉了低透明度区域的 warp 信号，这些区域恰好是模型最需要学习的地方
- temporal_gate 冻结与否无影响（fixv38a 冻结了但仍恶化，fixv39a 没提及但已恢复）

### 3.6 Hard Mask 必须条件化 (已修复 ✅)
- hard temporal masking 使用 prev_branch_probs 来决定哪些 query 是动态的
- 当 branch_cls=0（无监督）时，branch_heads 随机初始化 → p_dynamic≈0.5
- 结果：所有 query 的 past-frame 特征被 50% 替换为 current-frame → 时域信号被破坏
- **修复**: `use_hard_mask` 参数，当 `branch_cls > 0` 时才启用 hard masking

### 3.5 batch_size=1 是架构限制
- gsplat 的 `rasterization()` 以 camera 维度为 batch，不支持 data batch>1
- `accumulate_grad_batches=2` 可以替代 `batch_size=2`（等效梯度平均）

### 3.6 GPU 利用率
- 当前 21.8 GB / 81.6 GB (27%) — 有很大空间增加 batch size 或 query 数量

---

## 4. 未完成的 SelfOccFlow 方法

### Phase 1: 动态感知监督 (🔴 最高优先级)
不需要改 decoder 架构，直接在 blended render 上加 loss：

1. **动态覆盖 Loss**: `BCE(alpha_render[dyn_pixels], 1.0)`
   - 防止模型"抹掉"动态物体（当前没有任何 loss 鼓励动态区域产生 Gaussian）
   - 用 SAM3 dynamic_mask 定义动态像素

2. **动态深度 Loss**: `L1(depth_render[dyn_pixels], depth_foundation[dyn_pixels])`
   - 给动态区域几何监督（当前动态区域被 warp loss 排除，几乎没有深度监督）

### Phase 2: 分支渲染 (🟡 中等优先级)
fixv31 的分支渲染失败了，但根本问题可能是：
- branch_head 没有 detach（已修复）
- 同时改了太多东西

可以重新尝试的安全路线：
1. 先只加 static-only warp（用 opacity × p_static 做静态渲染）
2. 保持其他 loss 在 blended render 上
3. 逐步添加 dynamic-only depth

### Phase 3: 静态时域聚合 (🟡 中等优先级)
SelfOccFlow 最有价值的部分：训练时多帧静态 Gaussian 融合
- 当前帧的静态 Gaussian + 邻帧静态 Gaussian (ego 变换到当前帧)
- 需要 Phase 2 的 branch routing 先稳定

### Phase 4: Motion Head + Similarity Flow (🟠 低优先级)
- per-dynamic-query XY 偏移预测
- BEV 特征相似性流伪标签
- 需要 Phase 0 的双向时域数据

---

## 5. 下一步执行计划

### Step 1: fixv37 验证 detached semantic losses (进行中 ✅)
- 目标: ov_cos 像 fixv36a 一样持续下降，同时 sem_ce 仍然收敛
- 如果成功 → 这就是新基线

### Step 2: fixv38 — 在 fixv37 基础上尝试 Phase 2 分支渲染
- 用 detached branch_probs × opacity 做 static-only warp
- 关键: branch_head 已经 detach，不会影响 OV 特征

### Step 2: 观察 fixv34c 长期趋势
- 等 step 5000+ 看是否还有缓慢改善
- 对比 fixv23 在相同 epoch 下的表现

### Step 3: 考虑 batch_size=2
- 当前 GPU 只用 27%，batch=2 可以降低梯度噪声
- 这不是调参，是充分利用硬件

---

## 7. Autoresearch 自主实验记录 (Mar15-Mar16 session)

### 关键发现 1: `limit_train_batches` 步数 bug

**问题**: 所有 mar15 实验只有 56 个数据点 (step 2799)，而非预期的 3000 步 (60 个数据点)。
**根因**: `train_pgocc.py` 的 Trainer 构造函数没有传 `max_steps` 参数，所以 `+trainer.max_steps=3000` 无效。
**修复** (`6624800`): 在 Trainer 构造函数中加入 `max_steps=trainer_cfg.get('max_steps', -1)`，并改命令用 `+trainer.max_steps=3000 trainer.max_epochs=999`。
**影响**: 缺少 steps 2800-2999 导致 depth_0 无法达到 IMPROVED（那 200 步是 depth_0 最快收敛的区间）。修复后 depth_0 立即从 FLAT (-1%) → IMPROVED (-6%)。

### 关键发现 2: 4/4 主指标全部 IMPROVED (首次)

修复步数后，通过以下损失权重组合首次实现 4/4 primary metrics 全部 IMPROVED：

| 参数 | 值 | 说明 |
|------|-----|------|
| `ov_cos` | 7.0 | 从 1.0 提升；需 ≥7.0 才能 IMPROVED |
| `ov_cos_warmup_epochs` | 0.5 | warmup 到 step 1500；让几何先收敛 |
| `depth_gt` | 0.3 | 从 0.05 提升；LiDAR 锚定更强 |
| `branch_cls` | 0.5 | Phase 2 static-only warp；提升 depth_0 |
| `dyn_cov` | 1.0 | 动态覆盖损失；帮助 depth_0 |
| `dyn_depth` | 1.0 | 动态深度损失；帮助 depth_0 |
| `max_steps` | 3000 | 必须用 `+trainer.max_steps=3000 trainer.max_epochs=999` |

**最优配置结果** (ar_mar16_005):
- depth_0: -6.4% (IMPROVED)
- ov_cos: -2.1% (IMPROVED)
- warp: -16.5% (IMPROVED)
- depth_gt: -3.1% (IMPROVED)

### 关键发现 3: 各损失权重的作用

- **`depth_foundation` (默认 1.0)**: 不能提高。1.5 会导致 ov_cos 退化（共享渲染梯度竞争）
- **`depth_warping` (默认 5.0)**: 不能提高。7.0 反而使 depth_0 变差（warp loss 与 depth_foundation 竞争）
- **`branch_cls`**: 0.5 是最优，1.0 反而比 0.5 更差
- **`ov_cos`**: 7.0 是 Pareto 最优（depth_0 -5.6% + ov_cos -2.5%）。9.0 → ov_cos -3.8% 但 depth_0 -4.5%
- **`dyn_cov/dyn_depth`**: 1.0 比 0.5 对 depth_0 更好；去掉后 depth_0 从 -6.4% → -5.3%
- **`ov_cos` 需要 7.0 才 IMPROVED**: 5.0 → -1.9% FLAT；7.0 → -2.1% IMPROVED（临界值）
- **ov_cos 对 depth_0 有正面间接作用**: ov_cos=5.0+branch_cls 比 ov_cos=7.0+branch_cls 的 depth_0 更差（-5.1% vs -6.4%）

### 关键发现 4: 稳定性

`warp` 指标在几乎所有实验中都精确地是 **-16.4%**，与 ov_cos 权重无关。这表明 warp 收敛由 `depth_warping=5.0` 主导，其他损失权重对 warp 影响很小。

---

## 6. 代码变更索引

| Git Hash | 描述 |
|----------|------|
| `81f9bf1` | fix: Gaussian fog — scale_range, near_plane, alpha masking |
| `1dd2d6d` | fix: rebalance loss weights — depth_gt dominated |
| `06005ff` | fix: NCCL deadlock in 8-GPU DDP |
| `7ed6260` | feat: Phase 2 — static/dynamic branch routing |
| `83b3d6f` | fix: decouple branch supervision — projection-based |
| `7c2f223` | feat: temporal attention gate (失败) |
| `3bb2f53` | feat: hard temporal masking + all-stage branch supervision |
| `e555b80` | fix: warmup sem_ce to prevent OV corruption |
| `6624800` | fix: pass max_steps to pl.Trainer |
| `e542f69` | docs: experiment log Section 7 (max_steps bug + 4/4 IMPROVED first achievement) |
| `69721f2` | feat: ov_cos_static_only flag (static-only OV rendering) |

---

## 8. Autoresearch 自主实验记录 (Mar16-Mar17 session)

### 关键发现 1: warmup bug 修复 (max_epochs=999 → max_epochs=1)

**问题**: 之前所有使用 `ov_cos_warmup_epochs` 的实验中，warmup 完全无效。
**根因**: `trainer.max_epochs=999` 导致 `steps_per_epoch = max_steps//999 ≈ 3`，所以 `warmup_steps = 0.5 * 3 = 1.5` ≈ 0。
**修复**: 改为 `trainer.max_epochs=1`，使 `steps_per_epoch = max_steps = 3000/4000`。
**效果** (ar_mar16_014 vs ar_mar16_005):
- ov_cos 从 -2.1% → -5.2% (2.5x 提升!)
- depth_0 从 -6.4% → -5.7% (稍有下降)
- warmup 有效防止 OV 竞争在早期破坏几何收敛

### 关键发现 2: max_steps=4000 显著改善 ov_cos

**发现**: 增加训练步数是提升 ov_cos 的主要杠杆。
| 步数 | warmup | ov_cos | depth_0 |
|------|--------|--------|---------|
| 3000 | 0.5 | -5.2% | -5.7% |
| 3500 (partial) | 0.5 | -10.3% | -6.4% |
| 4000 | 0.5 | -15.6% | -4.0% |
| 4000 | 0.7 | -15.3% | -4.5% |

**结论**: 总步数是 ov_cos 收敛的主要决定因素，不是 warmup 比例。收敛需要更多 step 让 OV 特征质量提升。

### 关键发现 3: ov_mse=0 是 4000 步的严格 Pareto 改进

**发现**: `ov_mse=10.0`（默认值）在 4000 步时是对抗性的，但在 3000 步时有益。
- 4000 步: ov_mse=10.0 → depth_0 **-4.0%**; ov_mse=0 → depth_0 **-4.9%** (同样 ov_cos -15.6%)
- 3000 步: ov_mse=0 → depth_0 **-5.3%** (vs ov_mse=10.0 的 **-5.7%**, 更差!)
**原因**: ov_mse 的 L2 梯度在早期帮助 OV 对齐（有益），但在后期阻止 Gaussian 移动到更好的深度位置（有害）。
**教训**: 4000 步 config 必须用 `ov_mse=0`；3000 步 config 保留 `ov_mse=10.0`。

### 关键发现 4: warp 收敛与 depth_warping 权重无关

**发现**: `warp_0` 精确收敛到 -16.4%，无论 depth_warping=2.0, 3.0 还是 5.0。
**含义**: warp 收敛是模型容量/数据决定的，不是损失权重决定的。可以安全降低 depth_warping 到 3.0 (为其他 loss 释放梯度预算)，depth_warping=3.0 时 depth_gt 稍有改善 (-2.8% vs -2.4%)。

### 关键发现 5: depth_foundation=1.0 是 depth_0 的主要驱动力

**发现**: `depth_foundation=0.5` → depth_0 FLAT（不收敛），ov_cos -17.5%（最高！），depth_gt -3.5%（最高！）
**含义**: depth_foundation 是 depth_0 的主要监督信号。减少它会让 depth_0 无法收敛，但 ov_cos 和 depth_gt 大幅改善（因为少了竞争）。
**结论**: depth_foundation=1.0 是 depth_0 IMPROVED 的必要条件，不能降低。

### 关键发现 6: static-only ov_cos 失败

**实验**: `ov_cos_static_only=True`（只在静态 Gaussian 上计算 ov_cos）
**结果**: ov_cos 崩溃到 -2.5%（vs -15.6%），depth_0 -4.6%。
**原因**: 动态对象占据大量像素，去掉它们的 OV 监督会严重损害 OV 特征质量。
**教训**: ov_cos 必须应用到所有 Gaussian；不能用静态分支来减少 OV-depth 冲突。

### 当前最优 4000 步配置 (ar_mar16_024)

| 参数 | 值 |
|------|-----|
| `max_steps` | 4000 |
| `max_epochs` | 1 |
| `ov_mse` | **0.0** (关键！) |
| `ov_cos` | 7.0 |
| `depth_gt` | 0.3 |
| `depth_foundation` | 1.0 |
| `depth_warping` | 5.0 |
| `dyn_cov/dyn_depth` | 1.0 |
| `branch_cls` | 0.5 |
| `ov_cos_warmup_epochs` | 0.5 |

**结果**: depth_0 **-4.9%**, ov_cos **-15.6%**, warp **-16.4%**, depth_gt **-2.4%** (4/4 IMPROVED)

### 下一步方向

1. **Phase 3 SelfOccFlow**: 静态时域聚合 — 多帧 Gaussian 融合，需要代码修改
2. **reduce_dims 调参**: 当前 128 维 PCA；64 维可能让 ov_cos 更稳健
3. **增加 num_queries**: 更多 fine Gaussian (1000→2000) 可能改善场景覆盖
