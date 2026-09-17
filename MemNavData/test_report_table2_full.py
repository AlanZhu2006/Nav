"""Synthetic formatting fixtures; these are not navigation results."""
import json

import pytest

from MemNavData.report_table2_full import result_cells, render, digest


def group():
    return dict(queries=4, arms={"native":dict(successes=1,sr=.25,spl=.125),
        "cec":dict(successes=3,sr=.75,spl=.625)}, gain=2, loss=0,
        exact_mcnemar_p=.5, scene_cluster_bootstrap_ci95=[0.,1.])


def test_scores_keep_success_counts_and_spl_separate():
    cells=result_cells(group())
    assert cells == ["4", "1/4 (25.0%) / 0.125", "3/4 (75.0%) / 0.625", "+2/−0", "0.5", "[+0.0, +100.0] pp"]
    assert result_cells(None) == ["0", "—", "—", "—", "—", "—"]


def fixture_paths(tmp_path):
    a=dict(constructed=4,successes=3,mean_spl=.5)
    groups={s:{r:group() for r in ("novel","revisit")} for s in ("B","C")}
    summary=dict(completed=True,formal_result=True,plan_sha256="synthetic-test",
        source_scenes=2,source_budget=4,A=a,groups=groups,
        C_by_native_B_role={r:groups["C"] for r in ("novel","revisit")},
        supply={k:dict(queries=4,mean_geodesic_m=3.,distance_direction_cells={"2_to_4_front":4},support_sources={"A_only":4})
                for k in ("A_novel","B_novel","B_revisit","C_novel","C_revisit")},
        attrition=[dict(stage="A",reason="actual_A_failed")])
    path=tmp_path/"summary.json"
    path.write_text(json.dumps(summary))
    verification=tmp_path/"verification.json"
    verification.write_text(json.dumps(dict(verified=True,plan_sha256=summary["plan_sha256"],
        summary_sha256=digest(path),A=a,groups=groups,
        C_native_B_source_counts=dict(novel=1,revisit=1),verified_tasks=16)))
    return path,verification


def test_report_includes_all_five_rows_and_native_b_strata(tmp_path):
    report=render(*fixture_paths(tmp_path))
    for row in ("A / Novel","B / Novel","B / Revisit","C / Novel","C / Revisit"):
        assert row in report
    for row in ("Novel → Novel","Novel → Revisit","Revisit → Novel","Revisit → Revisit"):
        assert row in report
    assert "不是 GEM 自主连续完成三段的 joint SR" in report
    assert "actual_A_failed | 1" in report


def test_changed_or_unverified_summary_cannot_generate_final_report(tmp_path):
    path,verification=fixture_paths(tmp_path)
    data=json.loads(path.read_text())
    data["A"]["successes"]=4
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="matching independent verification"):
        render(path,verification)
