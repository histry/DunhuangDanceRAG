# Refiner Joint Evidence Synthesis

本阶段将已经完成的 Refiner 诊断、干预和报告修正结果合并为一份冻结证据账本。它是 report-level、read-only、CPU-only 的 JSON 合成，不构造模型、不加载 checkpoint、不执行 forward/backward/autograd、不创建 optimizer、不更新参数、不重新计算指标，也不搜索新的 case、alpha、阈值、架构或干预。

## 基线与输入

本实现必须以 `654671987d2fd41deac4fcb323adff49808e7574` 为父提交，并作为单独的 synthesis commit 推送到 `main`。所有输入都通过命令行显式传入，不搜索 latest artifact。

输入包括：

1. RCSP report 与其 reporting-logic review；
2. RCSP single-direction parameter attribution report；
3. single-direction decomposition report；
4. Phase 2 cross-width normalization report；
5. Phase 2.1 width-mechanism adjudication report；
6. BCTR report 与最新 BCTR reporting correction；
7. corrected SECDR report；
8. 历史 defective SECDR report，仅用于标记和排除，不参与证据计数、投票或最终结论。

每个输入 JSON 都会读取并记录 SHA-256，并验证其上游路径、SHA、schema、completed 状态、固定 cohort、read-only/production 状态和已冻结的科学分类。合成期间再次计算 SHA；任何输入变化都会 fail closed。输出目录必须是新的空目录，不能覆盖输入报告。

## 冻结科学边界

综合结果区分“机制证据”和“方法候选有效性”：

- RCSP：`PARTIAL_DIAGNOSTIC_ONLY`；角色条件化在 cross-event/width-10 出现 5/16 gate rescue，在 width-28 为 0/16，single-recording 为 0/32，不能作为正式方法候选。
- BCTR：保留 normalization 的 observational signal，但最新 correction 只修正报告逻辑；测量、decision input 和 scientific classification 不变，BCTR 不支持为充分干预。
- corrected SECDR：方向机制在 seen/new 上可被干预，但 width-28 temporal gate 为 0/16，gap shrink 和 endpoint/safety 条件不能组成正式候选；机制支持不等于解法支持。
- single-recording：保留 whole action、anatomy、temporal、anatomy×time、source/width shift、new-position/single/28 localized ascent 及 parameter-to-action bridge；不把局部冲突提升为唯一架构根因。

报告包含四个明确的 contradiction adjudications：normalization observation 与 BCTR failure、SECDR mechanism 与 efficacy failure、局部 block 与 whole cosine、连续 deficit 改善与 gate rescue 均标记为 `NOT_A_CONTRADICTION`。

## 最终判定树

唯一允许的最终结果为：

- `FORMAL_REFINER_METHOD_CANDIDATE_SUPPORTED`
- `MECHANISMS_IDENTIFIED_BUT_NO_SUFFICIENT_METHOD_CANDIDATE`
- `EVIDENCE_INTEGRITY_FAILURE`

正式候选必须同时满足 seen/new 的冻结 intervention support、endpoint non-degradation、safety/physical non-regression、自己的冻结证据支持和无 post-hoc selection。当前冻结输入的预期结果是 `MECHANISMS_IDENTIFIED_BUT_NO_SUFFICIENT_METHOD_CANDIDATE`，下一步是 `freeze_refiner_diagnostic_findings_and_stop_candidate_development`；不会授权 Pilot，也不会继续 unconstrained intervention search。

## 服务器执行

本项目按约定不在 Windows 本地执行代码验证。提交并推送后，在服务器执行以下命令；将 `<NEW_JOINT_SYNTHESIS_COMMIT>` 替换为本次 synthesis commit 的完整 SHA。

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail

git fetch origin main
git merge --ff-only origin/main

export PY=/home/disk/lsm/conda_envs/edge/bin/python
export ROOT=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915
export EXPECTED_COMMIT=<NEW_JOINT_SYNTHESIS_COMMIT>

export RCSP_REPORT="$ROOT/audits/role_conditioned_support_projection_20260902_132948_qu1hYg/result/report.json"
export RCSP_REVIEW="$ROOT/audits/role_conditioned_support_projection_20260902_132948_qu1hYg/result/reporting_logic_review_v1.json"
export PARAMETER_REPORT="$ROOT/audits/rcsp_single_direction_attribution_20260902_145442_VWA1LQ/result/report.json"
export SINGLE_REPORT="$ROOT/audits/single_direction_decomposition_20260902_213356_uH9fqu/result/report.json"
export PHASE2_REPORT="$ROOT/audits/cross_width_normalization_20260903_003257_14288/result/report.json"
export PHASE21_REPORT="$ROOT/audits/width_mechanism_adjudication_20260903_074314_6Vs6w5/result/report.json"
export BCTR_REPORT="$ROOT/interventions/bctr_temporal_reduction_20260903_092435_wPdK3U/result/report.json"
export BCTR_CORRECTION="$ROOT/audits/bctr_reporting_correction_20260903_121011_nkC2As/result/report.json"
export SECDR_REPORT="$ROOT/interventions/secdr_direction_rotation_corrected_20260903_121011_JByMYb/result/report.json"
export DEFECTIVE_SECDR_REPORT="$ROOT/interventions/secdr_direction_rotation_20260903_111428_bazm2q/result/report.json"

test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

"$PY" -m pytest -q tests/test_refiner_joint_evidence_synthesis.py

RUN_DIR="$(mktemp -d "$ROOT/audits/joint_evidence_synthesis_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
bash scripts/audit_refiner_joint_evidence_synthesis.sh \
  "$RCSP_REPORT" \
  "$RCSP_REVIEW" \
  "$PARAMETER_REPORT" \
  "$SINGLE_REPORT" \
  "$PHASE2_REPORT" \
  "$PHASE21_REPORT" \
  "$BCTR_REPORT" \
  "$BCTR_CORRECTION" \
  "$SECDR_REPORT" \
  "$DEFECTIVE_SECDR_REPORT" \
  "$RUN_DIR"

export REPORT="$RUN_DIR/result/report.json"
test -s "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert r["schema"] == "refiner_joint_evidence_synthesis_v1"
assert r["completed"] is True
assert r["provenance"]["implementation_parent_commit"] == "654671987d2fd41deac4fcb323adff49808e7574"
assert r["read_only"] is True
assert r["optimizer_steps"] == 0
assert r["model_loaded"] is False
assert r["forward_pass_performed"] is False
assert r["parameter_update_performed"] is False
assert r["lineage_verification"]["verified"] is True
assert r["lineage_verification"]["defective_secdr_excluded_from_scientific_evidence"] is True
assert r["final_decision"]["result"] in {
    "FORMAL_REFINER_METHOD_CANDIDATE_SUPPORTED",
    "MECHANISMS_IDENTIFIED_BUT_NO_SUFFICIENT_METHOD_CANDIDATE",
    "EVIDENCE_INTEGRITY_FAILURE",
}
assert r["pilot_allowed"] is False
print("JOINT_EVIDENCE_SYNTHESIS_REPORT_VERIFICATION_OK")
print("DECISION =", r["final_decision"]["result"])
print("NEXT_ACTION =", r["next_action"]["result"])
print("REPORT =", sys.argv[1])
PY

test -z "$(git status --porcelain)"
echo JOINT_EVIDENCE_SYNTHESIS_SERVER_AUDIT_OK
echo REPORT="$REPORT"
```

服务器审查时以 `result/report.json`、输入 SHA-256、`final_decision`、`claim_boundary` 和 `paper_safe_summary` 为准；代码存在、测试通过和报告生成本身都不能替代科学接受或 Pilot 授权。
