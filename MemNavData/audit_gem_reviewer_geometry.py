"""Offline bearing, historical age and optional support/offset analysis.

No learned model, simulator, or controller is called. Ground truth is used only
to score recorded outputs. Missing full-history metadata remains explicit.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from MemNavData.gem_bearing_attribution import ROOT, dump, load, sha

EVIDENCE = ROOT / ".diagnostics/gem_connected_memory_20260913/hpc_003/final"


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def bearing_error(position, yaw, goal_xz, bearing):
    delta = np.asarray(goal_xz) - np.asarray(position)[[0, 2]]
    truth = wrap(math.atan2(-delta[0], -delta[1]) - yaw)
    if bearing is None:
        return truth, None
    estimated = math.atan2(bearing[1], bearing[0])
    return truth, math.degrees(wrap(estimated-truth))


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0)
    assert np.isfinite(a).all()
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), maximum=float(a.max()), mean=float(a.mean()))


def support_metrics(history, query, anchor):
    poses = history["poses"]
    points = np.asarray([[p[k] for k in ("x", "y", "z")] for p in poses] + [history["end_position"]])
    curve = np.asarray(query["covis_curve"], dtype=float)
    assert len(curve) == len(poses)
    goal = np.asarray(query["floor_position"])
    # The first query adds its current RGB and evicts one of eight decision
    # observations. Current-view support is separate and not present in this curve.
    prior_fifo = history["decision_steps"][-7:]
    recent = float(curve[prior_fifo].max())
    old_domain = [i for i in range(8, len(curve)) if i not in prior_fifo]
    old = float(curve[old_domain].max())
    result = dict(history_frames=len(poses), all_historical_support=float(curve[8:].max()),
                  last_historical_observation_support=float(curve[-1]),
                  navdp_prior_fifo_indices=prior_fifo, navdp_prior_fifo_max_support=recent,
                  old_excluding_fifo_max_support=old,
                  old_supported_prior_fifo_unsupported=old >= .5 and recent < .1,
                  current_view_support=None,
                  current_view_support_note="The query endpoint is not one of the pre-action history observations",
                  raw_recent_max_support={str(n): float(curve[-n:].max()) for n in (16,32,64)})
    if anchor is not None:
        assert 8 <= anchor < len(poses)
        result.update(anchor=anchor, anchor_age=len(poses)-anchor,
                      anchor_goal_planar_offset_m=float(np.linalg.norm((points[anchor]-goal)[[0,2]])),
                      anchor_goal_3d_offset_m=float(np.linalg.norm(points[anchor]-goal)),
                      travel_since_anchor_planar_m=float(np.linalg.norm(np.diff(points[anchor:, [0,2]],axis=0),axis=1).sum()),
                      selected_anchor_support=float(curve[anchor]),
                      anchor_outside_controller_prior_fifo=anchor not in prior_fifo)
    return result


def audit(out, history_export=None, overlap_bounds=None):
    out.mkdir(parents=True, exist_ok=True)
    transfer, receipt = load(EVIDENCE / "transfer_verification.json"), load(EVIDENCE / "export_receipt.json")
    assert transfer["verified"]
    for path, digest in {**receipt["artifact_sha256"], **transfer["extra_sha256"]}.items():
        assert sha(EVIDENCE / path) == digest, path
    diagnosis, population = load(EVIDENCE / "first_query_diagnosis.json"), load(EVIDENCE / "population_description.json")
    assert diagnosis["complete"] and len(diagnosis["records"]) == 420
    assert diagnosis["plan_sha256"] == sha(EVIDENCE / "plan.json")
    pop = {r["index"]:r for r in population["rows"]}
    history = {}
    if history_export:
        history = {(r["cell"]["dataset"],r["cell"]["scene"],r["cell"]["episode"]):r
                   for r in load(history_export)["rows"]}
    rows = []
    for original in diagnosis["records"]:
        if original["mode"] not in ("legacy", "native_interval7"):
            continue
        r = dict(original)
        truth, signed = bearing_error(r["start_position"],r["start_yaw"],r["goal_xz"],r["first_query_bearing"])
        if signed is not None:
            assert abs(abs(signed)-r["first_query_bearing_error_deg"]) < 1e-5
        r.update(true_initial_bearing_deg=math.degrees(truth), initial_target_behind=abs(truth)>math.pi/2,
                 signed_bearing_error_deg=signed, absolute_bearing_error_deg=abs(signed) if signed is not None else None,
                 accepted_anchor_age=r["first_query_frame"]-r["first_query_anchor"] if r["first_query_accepted"] else None)
        assert r["first_query_frame"] == pop[r["index"]]["history_frames"]
        h = history.get((r["dataset"],r["scene"],r["episode"]))
        if h:
            assert h["trace_sha256"] == pop[r["index"]]["trace_sha256"]
            assert h["end_position"] == r["start_position"]
            q = next(q for q in h["queries"] if q["analysis_role"] == r["role"])
            assert np.array_equal(np.asarray(q["floor_position"])[[0,2]], r["goal_xz"])
            r["support_audit"] = support_metrics(h,q,r["first_query_anchor"] if r["first_query_accepted"] else None)
            if r["first_query_accepted"]:
                anchor_pose=h["poses"][r["first_query_anchor"]]
                anchor_xz=[anchor_pose["x"],anchor_pose["z"]]
                anchor_bearing,_=bearing_error(r["start_position"],r["start_yaw"],anchor_xz,None)
                r["historical_position_oracle_pose_error_deg"]=abs(math.degrees(wrap(anchor_bearing-truth)))
        rows.append(r)
    groups = {}
    for mode in ("legacy", "native_interval7"):
        for role in ("revisit", "novel"):
            group = [r for r in rows if r["mode"] == mode and r["role"] == role]
            accepted = [r for r in group if r["first_query_accepted"]]
            errors = [r["absolute_bearing_error_deg"] for r in accepted]
            ages = [r["accepted_anchor_age"] for r in accepted]
            groups[mode+"/"+role] = dict(queries=len(group),accepted=len(accepted),
                success=sum(r["reached"] for r in group), initial_target_behind=sum(r["initial_target_behind"] for r in group),
                bearing_error_deg=stats(errors), errors_above_30=sum(x>30 for x in errors),
                errors_above_90=sum(x>90 for x in errors), anchor_age=stats(ages),
                accepted_anchor_older_than_64_raw_frames=sum(x>64 for x in ages),
                complete_support_metadata=sum("support_audit" in r for r in group))

    # Link the archival geometry condition to the paper's NavDP result labels;
    # native_interval7 is a separate streaming-state diagnostic, not Table I.
    tablepath = ROOT / ".diagnostics/method_reconstruction_20260911/final-hpc-results/table1_summary.json"
    table = load(tablepath)
    table_index = {(r["dataset"],r["scene"],r["episode"],r["role"]):r
                   for r in table["paired_rows"] if r["controller"] == "navdp"}
    linked = []
    for r in rows:
        if r["mode"] != "legacy":
            continue
        t = table_index[(r["dataset"],r["scene"],r["episode"],r["role"])]
        assert t["cec"]["reached"] == r["reached"]
        linked.append(dict(dataset=r["dataset"],scene=r["scene"],episode=r["episode"],role=r["role"],
                           native_success=t["native"]["reached"],gem_success=r["reached"],
                           target_behind=r["initial_target_behind"],
                           explicit_turn_actions=t["cec"]["heading"]["turn_actions"]))
    result = dict(verified=True, scope="Initial-query geometry on the exact 70 main-table histories; no new navigation runs",
                  primary_condition="legacy reproduces the archived main-table NavDP outcomes",
                  secondary_condition="native_interval7 is a separate geometry-state comparison",
                  groups=groups, main_table_linked_queries=len(linked), linked_outcomes=linked, records=rows,
                  input_sha256={str(EVIDENCE / name):sha(EVIDENCE / name) for name in
                      ("first_query_diagnosis.json","population_description.json","plan.json","transfer_verification.json")},
                  pending=["Current-query image covisibility", "Recent-only versus full-history performance",
                           "Position-offset navigation comparison", "Long-range drift along full trajectories"],
                  source_sha256=sha(__file__))
    if history_export:
        result["history_export_sha256"] = sha(history_export)
        offset_groups=[]
        for lower,upper in ((0.,.5),(.5,1.),(1.,2.),(2.,math.inf)):
            subset=[r for r in rows if r["mode"]=="legacy" and r["role"]=="revisit" and r["first_query_accepted"]
                    and lower<=r["support_audit"]["anchor_goal_planar_offset_m"]<upper]
            offset_groups.append(dict(lower_m=lower,upper_m=upper if math.isfinite(upper) else None,queries=len(subset),
                gem_initial_error_deg=stats([r["absolute_bearing_error_deg"]for r in subset]),
                historical_position_oracle_pose_error_deg=stats([r["historical_position_oracle_pose_error_deg"]for r in subset]),
                gem_lower_error=sum(r["absolute_bearing_error_deg"]<r["historical_position_oracle_pose_error_deg"]for r in subset),
                gem_success=sum(r["reached"]for r in subset)))
        result["position_offset_analysis"]=dict(groups=offset_groups,
            comparator="Direction to the SAME selected historical camera position, using evaluator ground-truth poses",
            limitation="Offline oracle-pose diagnostic; not Raw retrieval, a deployable policy, or a new closed-loop success comparison")
    if overlap_bounds:
        overlaps=load(overlap_bounds)
        assert overlaps["verified"] and overlaps["history_export_sha256"]==sha(history_export)
        bounds={(r["dataset"],r["scene"],r["episode"]):r for r in overlaps["rows"]}
        subset_keys={k for k,r in bounds.items() if r["strict_old_memory_subset"]}
        cross_controller=[]
        for controller in ("navdp","vint","nomad"):
            subset=[r for r in table["paired_rows"] if r["controller"]==controller and r["role"]=="revisit"
                    and (r["dataset"],r["scene"],r["episode"])in subset_keys]
            wins=sum(r["cec"]["reached"]>r["native"]["reached"]for r in subset)
            losses=sum(r["cec"]["reached"]<r["native"]["reached"]for r in subset)
            n=wins+losses
            p=min(1.,2*sum(math.comb(n,k)for k in range(min(wins,losses)+1))/2**n) if n else 1.
            cross_controller.append(dict(controller=controller,queries=len(subset),native=sum(r["native"]["reached"]for r in subset),
                gem=sum(r["cec"]["reached"]for r in subset),paired_wins=wins,paired_losses=losses,unadjusted_exact_mcnemar_p=p))
        result["old_memory_subset"]=dict(queries=len(subset_keys),scenes=len({k[:2]for k in subset_keys}),
            criterion="History frame>=8 max covisibility>=0.5; last seven historical decision observations max<0.1; current view conservative covisibility upper bound<0.1",
            selection_uses_navigation_success=False,controllers=cross_controller,
            metadata=[bounds[k]for k in sorted(subset_keys)],overlap_bounds_sha256=sha(overlap_bounds),
            limitation="Post-hoc metadata-defined subset of the original main table; not an independent dataset or recent-only memory ablation",
            statistics_note="Descriptive small subgroup; the reported exact McNemar tests are unadjusted and do not model scene clustering")
        result["pending"].remove("Current-query image covisibility")
    dump(out / "geometry_audit.json", result)

    # Existing four-history pilot has complete metadata and complete trajectories.
    from MemNavData import table1_repaired_eval as existing
    parent = ROOT / ".diagnostics/gem_bearing_attribution_20260915/initial_turn_001"
    local = []
    for cell in load(parent / "plan.json")["cells"]:
        payload,frozen,_ = existing.history(cell)
        query = next(q for p in payload["pairs"] for q in p["queries"] if q["analysis_role"] == "revisit")
        trace = frozen["trace"]
        folder = parent / "evaluation" / cell["scene"] / "full_gem"
        plans = load(folder / "query_plans.json")["query_leg"]
        h = dict(poses=trace["poses"],end_position=trace["end_position"],decision_steps=[p["step"] for p in trace["plans"]])
        support = support_metrics(h,query,plans[0]["router_selected_anchor"])
        actual = [json.loads(line) for line in (folder / "full_plan_outputs.jsonl").read_text().splitlines()]
        assert len(actual) == len(plans)
        errors = []
        for p,a in zip(plans,actual):
            _,e = bearing_error(a["position"],a["yaw"],np.asarray(query["floor_position"])[[0,2]],p["memory_bearing_unit"])
            if e is not None:
                errors.append(abs(e))
        local.append(dict(scene=cell["scene"],support=support,accepted_plan_bearing_errors_deg=stats(errors),
                          definition="Per-history descriptive plan errors, not independent query samples"))
    dump(out / "four_history_support_audit.json",dict(histories=local,scope="Previously consumed four-history pilot"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.0),layout="constrained")
    for mode,label,color in [("legacy","Archived geometry","#286da8"),("native_interval7","Interval-7 geometry","#c15f27")]:
        accepted=[r for r in rows if r["mode"]==mode and r["role"]=="revisit" and r["first_query_accepted"]]
        e=np.sort([r["absolute_bearing_error_deg"] for r in accepted])
        axes[0].plot(e,np.arange(1,len(e)+1)/len(e),label=label,color=color)
        if mode=="legacy":
            axes[1].scatter([r["accepted_anchor_age"]for r in accepted],
                            [r["absolute_bearing_error_deg"]for r in accepted],s=18,alpha=.75,color=color)
    axes[0].set(xlabel="Initial bearing error (degrees)",ylabel="Fraction of accepted queries",ylim=(0,1.02))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Selected historical view age (raw frames)",ylabel="Initial bearing error (degrees)")
    for ax in axes:ax.grid(alpha=.2)
    for suffix in ("png","pdf"):
        fig.savefig(out / ("initial_bearing_audit."+suffix),dpi=180)
    plt.close(fig)
    g=groups["legacy/revisit"]
    text=["# 审稿补充：主表初始方向与记忆年龄审计", "",
          "已重新验证缓存元数据的哈希，并从目标真值、实际起点和输出方向独立计算角误差。",
          "这是已有运行的新增分析，不是新增导航成功率。legacy 条件与主表 NavDP 的逐查询成功标签全部一致。", "",
          "| 测量 | 结果 |", "|---|---:|",
          f"| Revisit 查询 / 接受召回 | {g['queries']} / {g['accepted']} |",
          f"| 初始方向误差中位数 / P90 | {g['bearing_error_deg']['median']:.2f}° / {g['bearing_error_deg']['p90']:.2f}° |",
          f"| 最大初始方向误差 | {g['bearing_error_deg']['maximum']:.2f}° |",
          f"| 超过 30° 的已接受方向 | {g['errors_above_30']} / {g['accepted']} |",
          f"| 选中历史帧年龄中位数 / 范围 | {g['anchor_age']['median']:.0f} / {g['anchor_age']['minimum']:.0f}–{g['anchor_age']['maximum']:.0f} 帧 |",
          f"| 年龄超过 64 帧 | {g['accepted_anchor_older_than_64_raw_frames']} / {g['accepted']} |",
          f"| 初始目标在后半平面 | {g['initial_target_behind']} / {g['queries']} |", "",
          "这能支持主表查询发布时的方向读出精度，不能推广成长期运动后的方向误差界。",
          "旧帧被选中不等于近期帧无法支持；必须结合近期和当前观测的共视判定。", "",
          "四条本机历史的已知情况：", "", "| 场景 | 选中帧年龄 | 历史—目标平面距离 | NavDP 近期七个历史观测的最大共视 |", "|---|---:|---:|---:|"]
    for r in local:
        s=r["support"]
        text.append(f"| {r['scene']} | {s['anchor_age']} | {s['anchor_goal_planar_offset_m']:.3f} m | {s['navdp_prior_fifo_max_support']:.3f} |")
    text += ["", "这四条均不满足预定的近期支持 < 0.1 条件，因此不能充当严格的旧记忆依赖子集。",
             "四条历史—目标距离均小于 1 m，因而也不足以证明较大位置偏移时 PnP 相对历史位置提示的价值。", ""]
    if history_export:
        text += ["主表中按所选历史视点—目标位置偏移分组：", "",
                 "| 平面偏移 | 查询数 | 指向历史位置：误差中位数 | GEM：误差中位数 |",
                 "|---|---:|---:|---:|"]
        for g in offset_groups:
            upper=g["upper_m"]
            label=f"[{g['lower_m']:g}, {upper:g}) m" if upper is not None else "≥2 m"
            if g["queries"]:
                text.append(f"| {label} | {g['queries']} | {g['historical_position_oracle_pose_error_deg']['median']:.2f}° | {g['gem_initial_error_deg']['median']:.2f}° |")
            else:text.append(f"| {label} | 0 | — | — |")
        text += ["", "这里的对照使用同一个经 GEM 选定的历史视点和真值位姿，目的是隔离‘返回历史位置’与‘定位目标’的差别。",
                 "它不是 Table IV 的 Raw retrieval，也没有执行新的对照导航。角度改善不能直接宣称成功率改善。", ""]
    if overlap_bounds:
        old=result["old_memory_subset"]
        text += [f"按预定支持条件筛出的主表子集：{old['queries']} 个查询、{old['scenes']} 个场景。", "",
                 "| 控制器 | Base | GEM | 配对增 / 损 |",
                 "|---|---:|---:|---:|"]
        for g in cross_controller:
            text.append(f"| {g['controller']} | {g['native']}/{g['queries']} | {g['gem']}/{g['queries']} | +{g['paired_wins']} / −{g['paired_losses']} |")
        text += ["", "筛选条件：旧历史共视≥0.5，NavDP 近期七个历史决策观测最大共视<0.1，且当前视图共视保守上界<0.1。",
                 "上界忽略遮挡并考虑深度 PNG 量化和饱和，13 例的上界均为 0；没有把最后一个历史观测冒充当前图像。",
                 "这是原始主表的事后分组分析，不是新独立任务；仍需 recent-only 检索对照才能直接比较两种记忆范围。", ""]
    (out / "RESULT.md").write_text("\n".join(text))
    print({k:v for k,v in result.items() if k in ("verified","groups","main_table_linked_queries")},flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--history-export",type=Path)
    parser.add_argument("--overlap-bounds",type=Path)
    args=parser.parse_args()
    audit(args.out.resolve(),args.history_export,args.overlap_bounds)
