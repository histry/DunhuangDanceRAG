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

曲率模型与角度无关，因此每次 SQP 迭代只构建一次，并由全部 12 个冻结角度
复用。二阶候选网格在 motion 所在 CUDA device 上生成、缓存和求值，最终的
词典序选择也在 device 上完成，不逐候选同步到 CPU。Projector 与完整 Guard
继续走原有权威实现，避免以性能优化为名改变验收合同。

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
