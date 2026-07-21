# SMPL24 与多帧率动作合同

本版本把骨架、旋转、时间单位和事件身份从隐式约定升级为运行时合同。迁移不兼容旧的混合资产；不要把 30 FPS 与 60 FPS 的缓存、Scheduler index、Duration checkpoint、V45 或 V46 checkpoint 交叉使用。

## 单一骨架与旋转入口

- `motion_geometry/smpl24.py` 是 SMPL24 关节名、父节点、米制 offset 和 EDGE151 通道的唯一来源。
- `motion_geometry/rotations.py` 是 column-concat Rot6D、SO(3) Log/Exp 和 near-pi 数值处理的唯一来源。
- 训练代码不再寻找旧 EDGE 工具，也不再提供 fallback offsets。
- 新 V44/V45/V46 checkpoint 写入 FPS、Rot6D、骨架 SHA-256 和物理导数单位；不匹配时立即拒绝加载。
- `V46_ALLOW_LEGACY_CHECKPOINT_CONTRACT=1` 只允许在 30 FPS 旧基线差分中临时读取旧权重，不能用于正式训练或 60 FPS 分支。

## AIST++ 适配

适配器读取 `smpl_poses`、`smpl_trans`、`smpl_scaling` 和源 FPS。AIST++ 缺少 FPS 字段时按其原始 60 FPS 处理。默认 `canonical_body` 策略使用统一 SMPL24 骨长，保留世界平移并把 `smpl_scaling` 写入报告；这样跨人物事件可拼接，同时没有假装消除体型差异。若公开数据预处理明确要求缩放平移，可选择 `scale_translation` 或 `inverse_scale_translation`，但同一实验只能使用一种策略。

## 分支构建

先预览，不执行：

```powershell
python scripts/build_multirate_branches.py `
  --source_dirs D:\datasets\ChangE D:\datasets\AIST++ `
  --output_root D:\DunhuangDanceRAG\outputs\multirate `
  --regression_audio D:\DunhuangDanceRAG\assets\music\test\audio\dunhuangwu2.wav `
  --router_ckpt_30 D:\weights\router_30.pt `
  --router_ckpt_60 D:\weights\router_60.pt `
  --planner_ckpt_30 D:\weights\planner_30.pt `
  --planner_ckpt_60 D:\weights\planner_60.pt `
  --duration_ckpt_30 D:\weights\duration_30.pt `
  --duration_ckpt_60 D:\weights\duration_60.pt
```

确认命令后执行并训练各自的 V45/V46：

```powershell
python scripts/build_multirate_branches.py `
  --source_dirs D:\datasets\ChangE D:\datasets\AIST++ `
  --output_root D:\DunhuangDanceRAG\outputs\multirate `
  --regression_audio D:\DunhuangDanceRAG\assets\music\test\audio\dunhuangwu2.wav `
  --router_ckpt_30 D:\weights\router_30.pt `
  --router_ckpt_60 D:\weights\router_60.pt `
  --planner_ckpt_30 D:\weights\planner_30.pt `
  --planner_ckpt_60 D:\weights\planner_60.pt `
  --duration_ckpt_30 D:\weights\duration_30.pt `
  --duration_ckpt_60 D:\weights\duration_60.pt `
  --execute --overwrite --train_v45_v46
```

构建顺序固定为：canonical 30 cache → canonical event intervals → 30 Event-DB/index/duration → 30 Router/Planner/Duration 资产包 → native 60 cache → 复用秒区间的 60 Event-DB/index/duration → 60 Router/Planner/Duration 资产包 → 各自训练 V45/V46。正式执行时三个 checkpoint 都必须声明与分支一致的 FPS；缺失或不一致会立即失败。

旧 Scheduler checkpoint 只可用于 30 FPS 的只读差分基线。运行 `scripts/run_no_training_regression.py` 时显式增加 `--allow_legacy_30fps_checkpoints`；该开关不能用于 60 FPS 或正式资产包。

## 三组消融

三组必须使用同一 WAV 和相同语义/路由策略：

1. `native30`：canonical 30 数据和 30 FPS checkpoint。
2. `native60`：独立 60 数据和重新训练的 60 FPS checkpoint。
3. `fps30_to_60`：第 1 组动作通过分通道 SO(3) 重采样到 60 FPS，重新计算接触，不训练新模型。

生成两组 native 结果后运行：

```powershell
python scripts/evaluate_multirate_ablation.py `
  --motion30 D:\results\native30.npy `
  --motion60 D:\results\native60.npy `
  --db30 D:\outputs\multirate\canonical30\event_db\events_aesd.npz `
  --db60 D:\outputs\multirate\native60\event_db\events_aesd.npz `
  --out_dir D:\outputs\multirate\ablation
```

报告以 m/s、m/s²、m/s³、rad/s 和 rad/s²比较三组，另外检查 30/60 Event-DB 的 UID 集是否相同。只有 UID、持续时间和音频合同一致时，30/60 对比才是有效的帧率消融。

## 必须重建的资产

改变骨架合同、Rot6D 合同、FPS 或切分时间后必须重建：

- retarget cache；
- Event-DB 与 AESD；
- Scheduler JSON/NPZ index；
- Duration index 和对应 FPS 的 Duration checkpoint；
- V45 与 V46 checkpoint。

Router/Planner 若输出或嵌入了帧数、transition length，也必须使用对应 FPS 的 checkpoint；只有纯秒制/语义 checkpoint 才允许在合同验证后复用。
