# V15.15g1f3 / V15.15h second-order composite

“稳定通过”只表示复合选择器完成 Projector 后，通过完整 transaction 的
authoritative Guard 闭包。二阶模型产生非零方向、二阶预测可行或有限半径
trial 改善，均不单独构成通过。

## 冻结合同

V15.15g1f3 不训练 Adapter，也不扩充教师。以下项目沿用并冻结：Adapter、
教师 bank、train conformal 包络、2/3/5 步预算、12 级角搜索、`1e-4`
ownership 半径、EDGE151 Projector、完整 Guard 和 `0.03` 修复门槛。

新增数值核位于 `training/refiner_v15_15g1f3_second_order.py`。它对标量
测地线角度沿 `geodesic -> product retraction -> FK -> metric` 的真实路径
做 float64 一、二阶求导，因此二阶项包含球面测地线加速度。低维基由
ownership 球面切空间中的 shadow、endpoint、temporal 物理协向量构造；
只通过方向 HvP 和 polarization 形成最多 `3 x 3` 的模型，不形成环境维度
Hessian。max/p95 活跃集只在该角度的预测模型内冻结，真实 trial 会重算硬
指标和完整 Guard。

扩展点的 Guard 活跃项在 `theta=0` 记录后，会显式传入全部 float64 曲率
求值（包括 HvP 恢复的正负 epsilon 路径）；epsilon 路径观察到的硬活跃集
变化不会改写预测合同。只有随后真实有限半径 trial 的重新计算结果具有权威
验收效力，变化会报告为 `active_set_transition_model_mismatch`。

曲率模型与角度无关，因此每次 SQP 迭代只构建一次，并由全部 12 个冻结角度
复用。仅当网格没有共同可行点时，每个角度才以全部网格方向（至多 768 个）
为起点，在低维单位球面上执行固定 64 次连续 Riemannian 联合 SQP 精化；约束
值、解析梯度、固定角度线搜索及最终
词典序选择都在 motion 所在 CUDA device 上完成。网格不再作为“无共同可行
方向”的最终判据。三个约束按各自有符号安全边界 gap 归一化，不能再用指标
动态范围掩盖正的 Guard 违反。Projector 与完整 Guard 继续走原有权威实现。

endpoint/temporal 的有符号有限 gap 按
`current_delta + strict_limit + safety_margin` 计算。已经安全进入严格通过域的
项可在不越过安全边界的前提下使用余量，不再人为要求每一步继续下降；最终
是否通过仍由真实 trial 和完整 Guard 决定。

`k=2/3/5` 是真实多步预算：每一迭代把当前 endpoint、temporal 和最坏完整
Guard shadow 到安全闭包边界的剩余有符号 gap 除以剩余步数。真实 trial 必须
完成这一步三项联合进度；最后一步必须进入 endpoint/temporal 严格通过域且
Guard shadow 不大于零。若提前达到真实 raw 闭包则停止该预算，随后仍须经过
复合 selector、Projector 和完整 transaction Guard 才能称为稳定通过。
若某一中间步的三项等分配额在冻结二阶模型中不可行，求解器显式进入
restoration/filter 子问题，并从第一步起按完整剩余闭包 gap 重新执行连续
SQP，选择面向最终闭包的最小联合归一化残差方向，而不是继续围绕本步等分
配额排序。真实 trial 只有在
不越过已经安全的边界且正的联合闭包缺口严格下降时才可继续；最后一步不接受
filter 进展代替完整闭包，也不存在隐式一阶或白名单 fallback。

每个二阶扩展点把距离当前 hard Guard 最大 shadow 不超过冻结 `1e-3` band
的全部 Guard 项组成 transition bundle；该 bundle 的具体项在本次曲率构建和
12 级角搜索中冻结，并以固定温度 logsumexp 构造 shadow 二阶模型。真实 trial
仍重新计算全部 hard Guard 和活跃集。这样允许模型预见临界项接管，同时不把
真实 Guard 替换成平滑代理。

若保守 wake gate 使 Adapter 候选方向严格为零，二阶模块可使用同一冻结
Adapter 解码器的门控前方向建立 `1e-4` 球面起点。这个方向仍只由运行时
observable 产生，不读取 teacher、split、single/cross 标签或案例 UID；门控
后的 Adapter 候选身份不变，也不能因此跳过 Projector 或完整 Guard。门控前
方向仍为零时继续 `zero_gradient_abstention`，不制造教师。

为避免重复曲率计算，二阶候选只对 observable 已激活的 transaction 生成；
train 中额外声明的目标和 exact cross 行仍按既有离线校准规则执行。某候选一旦
通过 raw、Projector 与完整 Guard 审计，其 tangent 可在更大的 `k` 预算中直接
复用并再次接受权威审计，不再重建相同的 float64 Hessian/HvP。Adapter incumbent
同样直接锁定。该缓存不跨冷启动进程，三次独立复跑要求不变。

train 校准行使用包络中已经冻结的 leave-one-transaction-out observable
分数；禁止再用包含该行的最终模型给该行做 in-sample 判定。此规则不读取
fold 中保存的 offline label。development、一次性 held-out 和整曲推断仍只
使用 train 冻结的 final models，因而不会把 train UID 或案例白名单带入运行时。

五个已声明的 g1f2 mismatch UID 在 `train_calibration` 角色中作为离线数值探针
强制进入候选生成、Projector 与完整 Guard 闭包审计，因为它们不是现有
conformal 包络的 cross 拟合成员。该探针集合只用于校准验收，不写入 V15.15h
运行时激活逻辑；development、held-out 和整曲均不能用这些 UID 激活候选。
同理，train bank 中已有 `exact_projected_direction` 行作为离线 cross 校准行
必须全部进入 train selector 非回归审计，避免 conformal uncertainty 让冻结
bank 的 required projected count 无法验证；该离线角色也不进入任何运行时入口。

若某个基方向的方向 HvP 非有限，该方向会在建模前被确定性排除；若组合方向
失败，则排除索引较后的基向量并重新构建已验证子空间。模型不会消费 NaN，
也不会用零值伪造曲率；没有任何可验证方向时仍以
`nonfinite_or_unverified_curvature` fail-closed。该规则不读取案例 ID，并由
train 冻结合约锁定后原样用于 development、held-out 和整曲推断。

仅当扩展点的一阶值有限、autograd 二阶反传因零范数支路非有限时，数值核会
在真实路径上用冻结的 `1e-4/3e-4` 角半径计算 `F'(+eps)` 与 `F'(-eps)`，
恢复方向 HvP。两级结果必须通过固定的相对/绝对一致性检查；不一致的方向按
上述规则排除，而不是隐式 fallback 或放宽 Guard。

状态集合固定为：

- `second_order_closure_succeeded`
- `insufficient_second_order_predicted_progress`
- `second_order_finite_radius_model_mismatch`
- `active_set_transition_model_mismatch`
- `nonfinite_or_unverified_curvature`
- `second_order_solver_failure`

case 42 继续作为方向导数不一致通道；case 131 的零梯度继续 abstain。五个
train 校准目标只用于验收，未进入推断白名单。

## 服务器执行顺序

首先拉取提交并设置完整 commit：

```bash
git pull --ff-only origin main
export EXPECTED_COMMIT=$(git rev-parse HEAD)
```

运行三次独立冷启动 train 校准和一次 reused-development。任一 train 未恢复
五案例、丢失 Adapter incumbent、未满足分组数量，或三次选择/状态/指标不
一致，脚本都会在进入 development 前失败。development 要求原八个
`cross_short` 和 case 53 `cross_long` 均闭包，随后以 exclusive-create
方式生成 `g1f3_frozen_contract.json`。

```bash
bash scripts/run_refiner_v15_15g1f3_train_dev_server.sh
```

从一个从未进入 g1b--g1f3 的候选 bank 中，先封存 transaction 和 manifest：

```bash
export FINAL_HELD_OUT_CANDIDATE_BANK=/absolute/path/to/untouched_candidates.pt
bash scripts/prepare_refiner_v15_15g1f3_final_held_out_server.sh
```

只对 `outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_BANK` 指向的封存 bank 独立
运行 Oracle。Oracle JSON 必须明确包含匹配的 `transaction_id`、
`cross_long_feasible: true`，以及非空的 `raw_evidence`、
`projector_evidence`、`scope_evidence`、`full_guard_evidence`。然后才可消费
held-out：

```bash
export FINAL_HELD_OUT_ORACLE_REPORT=/absolute/path/to/oracle.report.json
bash scripts/run_refiner_v15_15g1f3_final_held_out_server.sh
```

一次性 receipt 会在 g1f3 启动前创建。失败 transaction 自动留下
`FAILED_HELD_OUT_BECOMES_DEVELOPMENT_EVIDENCE` 标记；禁止修复后重跑并继续
称为 held-out，必须重新封存另一个未查看 transaction。

三道闸门通过后打包：

```bash
export BASE_REFINER_CHECKPOINT=/absolute/path/to/diagnostic_state.pt
bash scripts/package_refiner_v15_15h_composite_server.sh
```

产物固定命名为：

```text
v15_15h_adapter_second_order_composite.pt
v15_15h_adapter_second_order_composite.contract.json
```

它是 “Adapter + second-order repair composite”，不是重新训练得到的单一
Adapter checkpoint。JSON 合同锁定 base Refiner、Adapter、conformal、g1f3、
train/dev/held-out manifest、三阶段报告、代码提交和全部推断常量的 SHA256。

## 整曲推断

设置以下两个变量后，`scripts/pipeline.sh` 会把复合模型和合同一起传给
`routing/boundary_closed_loop.py`：

```bash
export REFINER_COMPOSITE_MODEL=$(cat outputs/LATEST_REFINER_V15_15H_COMPOSITE_MODEL)
export REFINER_COMPOSITE_CONTRACT=$(cat outputs/LATEST_REFINER_V15_15H_COMPOSITE_CONTRACT)
bash scripts/pipeline.sh
```

每个完整 motion transaction 执行：observable conformal 判定、Adapter d0、
incumbent 完整 Guard 锁定、g1f3 的 2→3→5 二阶 SQP、Projector、完整
transaction Guard、原子提交或 identity。推断不读取 `single/cross`、教师或
案例 ID；不确定、全部预算失败、非有限曲率或求解失败均返回 identity 并在
报告保留原因。提交后再次审计完整 transaction。

原整曲 physical、boundary continuity 和 activity 最终门保持 fail-closed。
失败时入口只写 `.rejected.npy` 和诊断 JSON，抛出错误后 pipeline 不会进入
固定机位 MP4 渲染。
