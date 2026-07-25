# 无配对音乐语义最优传输接地

## 研究边界

本实现用于不存在真实音乐—敦煌舞动作配对标注的低资源场景。系统复用历史 Router 的 `music_encoder` 作为音乐域先验，通过结构乐句、MSSD 概率、校准 AESD 概率和稀疏熵正则最优传输建立软监督。

该路线不提供音乐帧与动作帧的真实同步真值，因此：

- `supervision=semantic_ot_teacher`；
- `is_ground_truth_pair=false`；
- 不允许写成 `dataset_paired`；
- 教师 Top-1 一致率不是人工配对检索准确率；
- 论文中应使用“弱监督音乐—动作语义接地”或“无配对语义最优传输对齐”。

## 数据流

```text
整首训练音乐
  -> 按歌曲划分 train / validation / test
  -> 结构新颖度与节拍吸附的可变长乐句
  -> 历史 music_encoder + 八类软语义头
  -> MSSD 概率、CLAP、64 帧时序特征

源隔离 Event-DB
  -> 分割后独立构建 AESD
  -> 分组证据融合、先验校正、熵与 Top-2 间隔
  -> intrinsic transition prior

MSSD × AESD
  -> JS 语义代价 + 时长 + 质量 + 置信度 + 内在风险 + 来源平衡
  -> 稀疏 Sinkhorn
  -> 每乐句 Top-K 多正样本权重
  -> Mixed-Curvature Grounder 软双向蒸馏
```

## 无泄漏合同

1. 音乐按整首歌曲先划分；同一歌曲的所有乐句只能属于一个集合。
2. 动作按 `source_uid` 先划分，再切 Event。
3. 分别构建 `music_train × motion_train`、`music_validation × motion_val` 和 `music_test × motion_test`。
4. Grounder 直接接收独立 train/validation 数据集，不从伪配对身份图随机切分。
5. 原始 `DB_SPLIT_ROOT` 不被覆盖；校准 AESD 与嵌入结果保存在本次 `semantic_ot` 运行目录。

## AESD 校准

现有代码中的 `aerial_curve -> lyrical_flow` 被修正为 `aerial_curve -> aerial_curve`。语义证据分为五组并在组内归一化：

- 显式对齐 0.25；
- 舞蹈类别 0.20；
- Event family 0.25；
- 动态属性 0.20；
- 底层描述子 0.10。

输出同时保存原始概率、校准概率、主类别、次类别、归一化熵、Top-2 margin 和 ambiguity 标志。类别先验修正是轻量校准，不强制八类均匀。

## 转移风险

AESD 的事件内入口—出口量被命名为 `intrinsic_transition_prior`，只用于轻量预筛。运行时真实决策使用 `contracts.boundary.transition_multiscale_risk` 的事件对物理测量：

- hard reject：重新路由；
- 低风险：直接连接；
- 中风险：SO(3)/根节点几何对齐；
- 高风险：对齐后复测；残余风险仍高时才进行 Contact-Guided Masked Inpainting。

## 科研评价

没有人工配对真值时，不报告配对检索 Accuracy/R@K。建议报告：

- MSSD–AESD 加权 JS 散度；
- Sinkhorn residual 与 transport cost；
- 语义软覆盖、来源覆盖与 family 覆盖；
- AESD 熵、Top-2 margin 与 ambiguity；
- hard reject rate、局部修复触发率、修复后风险；
- Peak Jerk、Exit Acceleration、contact slip、root velocity gap 与 warp ratio。
