# 小论文二：Fisher–Rao 离散 Graph-SB 整曲 Event 路由

## 1. 实现边界

本模块优化的是整曲离散 Event 路由，不修改 V45/V46 的局部动作生成器，也不
把 Event ID 当作连续姿态变量。最终决策对象是每个音乐 slot 的候选概率：

\[
p_t\in\operatorname{int}\Delta^{K_t-1},
\]

其 Fisher–Rao 距离为：

\[
d_{\mathrm{FR}}(p,q)
=2\arccos\sum_i\sqrt{p_iq_i}.
\]

在时间展开 Event 图上构造参考 Markov 核：

\[
Q_t(i,j)\propto
M_t(i,j)\exp\left[-C_t(i,j)/\varepsilon\right],
\]

其中 `M` 是不可放宽的硬可行性掩码，`C` 包含 SO(3) 乘积端点距离、
Lorentz 层级距离和现有物理边代价。多边际 IPF 求解与参考路径测度 KL
距离最小、同时匹配各 slot Fisher–Rao 候选边际的路径测度。

这是一种有限离散状态空间的多边际 Schrödinger 路由，不声称实现连续 SDE
Schrödinger Bridge，也不声称当前已训练 GSBoG 式 CTMC 控制策略。

## 2. 分层代码

- `routing/fisher_rao.py`
  - 概率归一化；
  - 带结构零的 simplex softmax；
  - Fisher–Rao 距离和测地中点；
  - categorical entropy 与 KL。
- `routing/graph_schrodinger.py`
  - 硬可行参考 Markov 核；
  - 精确 forward–backward；
  - 多边际 IPF；
  - 后验边转移；
  - Viterbi MAP 路径。
- `routing/event_graph_geometry.py`
  - Event 节点 Anatomy/Heading 不可变门；
  - \(SO(3)^{24}\) 端点乘积距离；
  - 论文一 Lorentz 因子距离；
  - 物理边代价和硬边支持。
- `routing/global_path.py`
  - 时间展开候选图；
  - Fisher–Rao 目标边际；
  - Graph-SB 求解；
  - history-dependent diversity 校验；
  - 受约束后验解码和可审计 legacy 回退。

## 3. 第零层：重建 SO(3) 路由端点

重新运行 Event 内蕴几何增强：

```powershell
python -m events.intrinsic_geometry `
  --db D:\path\events.npz `
  --fps 30
```

新增字段：

- `v46_55_route_geometry_schema_version`；
- `v46_55_entry_rotation_matrix`，形状 `[N,24,3,3]`；
- `v46_55_exit_rotation_matrix`，形状 `[N,24,3,3]`。

字段来自投影后的合法 SO(3) 矩阵，不直接比较未约束 Rot6D 通道。历史
`v46_53_geometry_schema_version` 和 112 维 Grounder 描述符保持不变，因此
不会使已有 Grounder checkpoint 因新增路由字段失效。

## 4. 第一层：可选 Lorentz 边因子

若要完成推荐的完整边代价，先使用小论文一混合曲率 Grounder 嵌入 Event-DB：

```powershell
python -m grounding.mixed_curvature embed `
  --db D:\path\events.npz `
  --checkpoint D:\path\v46_53_mixed_curvature_grounder.pt
```

随后设置：

```powershell
$env:V46_55_REQUIRE_LORENTZ_EDGE = "1"
```

如果仍在使用 legacy Grounder，应保持为 `0`。代码会在报告中记录 Lorentz
边覆盖率，不会将欧氏 embedding 伪装成 Lorentz 点。

## 5. 第二层：启用 Graph-SB

可参考 `configs/fisher_rao_graph_sb.env.example`：

```powershell
$env:V46_55_ROUTE_SOLVER = "fisher_rao_graph_sb"
$env:V46_55_REQUIRE_SO3_EDGE = "1"
$env:V46_55_SB_ALLOW_LEGACY_FALLBACK = "1"
```

正常运行现有：

```powershell
python -m routing.closed_loop generate ...
```

最终报告同时保留兼容字段 `v46_53_global_route`，并新增
`v46_55_graph_sb_route`，其中包括：

- IPF 迭代数、最大 L1 和 Fisher–Rao 边际残差；
- 路径熵和 MAP log probability；
- 每层候选目标/后验概率；
- SO(3) 与 Lorentz 边覆盖率；
- 硬边删除原因；
- Viterbi 或 history-constrained decoder 类型；
- 是否发生 fallback 及完整原因。

## 6. 两种运行策略

### 生成安全策略

```powershell
$env:V46_55_SB_ALLOW_LEGACY_FALLBACK = "1"
```

IPF 不收敛、图出现死路或历史约束耗尽时，回退到原有全局 beam。回退不是
静默行为：顶层报告 schema 保持为
`v46_55_fisher_rao_graph_sb_fallback_v1`，同时写入兼容字段
`v46_53_global_route` 和专用字段 `v46_55_graph_sb_route`。其中保留
`requested_solver`、完整失败原因、Graph-SB 尝试记录以及嵌套的 legacy 路由
报告，避免 fallback 后被 V46.53 schema 覆盖。

### 论文严格评估策略

```powershell
$env:V46_55_SB_ALLOW_LEGACY_FALLBACK = "0"
$env:V46_55_REQUIRE_SO3_EDGE = "1"
$env:V46_55_REQUIRE_LORENTZ_EDGE = "1"
```

严格实验必须 fail closed，不能把 Graph-SB 与 legacy beam 的结果混在同一
方法组中。若希望报告 fallback 性能，应作为单独的 safety hybrid 方法。

## 7. 历史相关约束

source 连续次数、family/source 全局占比和 Event cooldown 不仅依赖相邻
Event，不能完全编码成普通 pairwise Markov 边。本实现采用两层处理：

1. 相邻重复、Anatomy、Heading 和严重物理约束直接成为结构零边；
2. Viterbi 路径若违反长历史约束，则在同一个 Graph-SB 后验上运行
   history-constrained beam decoder。

报告会明确记录解码器类型，不会把第二种情况误报为普通 Event-ID 状态上的
精确 Viterbi。现有下游 Heading/Anatomy/Physics 模拟器仍是最终权威提交门。

## 8. 测试

轻量数学与集成测试：

```powershell
python -m unittest -v `
  tests.test_fisher_rao_graph_sb `
  tests.test_event_graph_geometry `
  tests.test_global_path_graph_sb_integration `
  tests.test_event_geometry `
  tests.test_routing_diversity
```

正式实验还应分别比较 local Top-1、legacy beam、Viterbi、ILP、
entropy-regularized CRF 和 Graph-SB，并报告路径效用、硬约束违反率、最终
生成成功率、最优性差距、ECE/Brier/AURC、时间和内存。

## 9. 当前科研边界

当前 target marginal 仍由 Grounder unary、质量、Anatomy 和 rank prior
构成；Graph-SB 路径测度已经是严格实现，但失败概率尚未由真实 assembly
结果校准。因此现阶段可以声称：

> Fisher–Rao categorical marginals and multi-marginal discrete
> Schrödinger routing on a hard-feasible time-expanded Event graph.

尚不能声称：

- learned calibrated failure posterior；
- learned CTMC control policy；
- continuous Schrödinger Bridge；
- 全部 history constraint 都由未扩展的 Event-ID Markov 状态精确表示。
