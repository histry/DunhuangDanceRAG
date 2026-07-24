# 小论文一：混合曲率 Gaussian-Wasserstein Grounder

## 1. 设计边界

本模块只负责音乐—动作事件检索，不修改 V45/V46/V53 的 EDGE-151 物理动作
流形。最终隐空间为：

\[
\mathbb H_\kappa^{d_h}\times
\mathbb S^{d_s-1}\times
\prod_{b=1}^{5}\mathcal G_{\mathrm{BW}}^{8}\times
\mathbb R^4 .
\]

- Lorentz：事件族、舞姿和动作层级；
- Sphere：归一化 CLAP 跨模态语义；
- Gaussian-BW：五个不重叠身体部位的 SO(3) 切空间动力学均值与协方差；
- Euclidean：时长、质量、语义置信度和事件位置；
- uncertainty：数据质量与跨域不确定性，不作为查询相关的度量权重。

## 2. 分层执行

### 第零层：生成身体部位 Gaussian 字段

先在关闭 Grounder 训练的情况下重建 Event-DB。`events/intrinsic_geometry.py`
会新增：

- `v46_53_bodypart_gaussian_mean`，形状 `[N,5,8]`；
- `v46_53_bodypart_gaussian_covariance`，形状 `[N,5,8,8]`；
- `v46_53_bodypart_gaussian_samples`，形状 `[N,5]`。

协方差经过对角收缩和最小特征值截断，短事件不会生成奇异 SPD。

### 第一层：准备真实音频配对清单

复制 `configs/mixed_grounding_pairs.example.json`，每行必须提供稳定
`event_uid`、真实 `audio_path`、`start_sec` 和 `end_sec`。相同音乐片段对应
多个合理动作时使用相同 `pair_id`，训练会按多正样本处理。

不得把 Router 的规则 Top-K 结果标记为 `dataset_paired`；若用于弱监督实验，
应明确写成 `heuristic` 并单独报告。

### 第二层：构建配对数据

```powershell
python -m grounding.paired_data `
  --event_db D:\path\train\events.npz `
  --manifest D:\path\mixed_pairs.json `
  --out D:\path\paired_grounding.npz `
  --model_name clap `
  --cache_dir D:\path\clap_cache
```

构建器要求真实 CLAP 成功率为 100%，不会使用固定随机投影，也不会回退到规则
语义。输出同时包含64帧时序音乐特征、动作几何、Gaussian-BW 目标、来源和层级
标签。

### 第三层：训练

```powershell
python -m grounding.mixed_curvature train `
  --data D:\path\paired_grounding.npz `
  --out D:\path\v46_53_mixed_curvature_grounder.pt `
  --epochs 120 `
  --batch_size 96 `
  --seed 20260724
```

训练先以 `source_id`、`pair_id` 和音频文件/显式音频组构造身份图，再按其
连通分量划分训练/验证集。任一身份都不能跨越两侧；若整个数据只有一个连通
分量，训练会直接失败，避免以泄漏验证结果冒充泛化性能。checkpoint 保存训练
集统计量、事件库身份合同、可学习曲率、全局乘积权重和损失配置。报告包含双向
R@1/5/10、MRR 和 mAP，并记录两侧的来源、配对和音频组。

### 第四层：嵌入 Event-DB

```powershell
python -m grounding.mixed_curvature embed `
  --db D:\path\test\events.npz `
  --checkpoint D:\path\v46_53_mixed_curvature_grounder.pt
```

推理只使用 checkpoint 中的训练来源统计量，不读取验证/测试自己的均值和方差。
通过 `events.build_pipeline` 训练 mixed Grounder 时，训练 Event-DB 会在
checkpoint 生成后立即执行同一嵌入步骤，写入
`v46_53_mixed_lorentz` 等全部乘积流形因子；验证/测试库继续使用该训练
checkpoint 嵌入。

### 第五层：接入完整构建链

设置：

```powershell
$env:V46_53_GROUNDER_ARCHITECTURE = "mixed"
$env:V46_53_GROUNDER_PAIRED_DATASET = "D:\path\paired_grounding.npz"
```

随后运行原 Event-DB 构建链。混合 checkpoint 使用
`v46_53_mixed_curvature_grounder.pt`；历史
`v46_53_dual_branch_grounder.pt` 仍按原逻辑加载。

运行时 slot 必须包含：

- `clap_embedding`：与训练模型相同维数的未投影 CLAP；
- `temporal_features`：任意帧数的12维时序音乐特征。

`scripts/pipeline.sh` 在 mixed 模式下会于闭环生成前自动执行
`grounding.audio_query`，并用 checkpoint 校验 CLAP/时序维度，再将补齐后的
schedule 交给 Router。科研配置默认
`V46_53_MIXED_REQUIRE_RUNTIME_AUDIO=1`：缺少任一字段时直接失败，禁止静默
回退。只有显式关闭该严格开关时，兼容路径才会回退到确定性 AESD 分数。

可用以下命令为现有 schedule 一次性补齐字段：

```powershell
python -m grounding.audio_query `
  --audio D:\path\query.wav `
  --schedule D:\path\schedule.json `
  --out D:\path\schedule.mixed.json `
  --checkpoint D:\path\v46_53_mixed_curvature_grounder.pt `
  --cache_dir D:\path\clap_cache
```

## 3. 逻辑安全检查

```powershell
python -m unittest -v `
  tests.test_grounding_manifold_ops `
  tests.test_bodypart_gaussian_geometry `
  tests.test_paired_grounding_data `
  tests.test_mixed_grounding_contracts `
  tests.test_mixed_grounding_torch
```

无 PyTorch 的轻量环境会跳过 GPU/反向传播测试，但 NumPy 几何、SPD、数据合同
和来源隔离测试仍会执行。正式训练环境必须运行全部测试。
