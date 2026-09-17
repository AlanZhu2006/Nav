"""Render the completed, independently verified Table-II experiment in Chinese."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_cells(group):
    if group is None:
        return ["0", "—", "—", "—", "—", "—"]
    n = group["queries"]
    arms = group["arms"]
    scores = [f'{arms[a]["successes"]}/{n} ({100*arms[a]["sr"]:.1f}%) / {arms[a]["spl"]:.3f}'
              for a in ("native", "cec")]
    lo, hi = group["scene_cluster_bootstrap_ci95"]
    return [str(n), *scores, f'+{group["gain"]}/−{group["loss"]}',
            f'{group["exact_mcnemar_p"]:.6g}', f'[{100*lo:+.1f}, {100*hi:+.1f}] pp']


def render(summary_path, verification_path):
    summary, verified = load(summary_path), load(verification_path)
    if not (summary["completed"] and summary["formal_result"] and verified["verified"]
            and verified["summary_sha256"] == digest(summary_path)
            and verified["plan_sha256"] == summary["plan_sha256"]):
        raise ValueError("A completed summary and matching independent verification are required")
    lines = ["# Table II：实际单目历史上的逐阶段探索与回访", "",
        "本报告由完整汇总和原始轨迹独立复算生成。A 为真实单目 NavDP 导航；"
        "B/C 对同一查询执行 native/GEM 配对，运行时不读取角色标签。"
        "C 以成功的 native-B 实际历史为参考，前序 Novel-B / Revisit-B 各占一半。", "",
        "这是阶段条件的共享前缀实验，不是 GEM 自主连续完成三段的 joint SR。"
        "场景为已使用过的 MP3D/PT1 资产，不称为新 held-out confirmation。", "",
        "## 1. 主结果", "",
        "SR 的分母为实际构造并完整评测的查询数。SPL 用真实逐动作位移及末步落点积分，包含失败查询的零分。"
        "配对检验为双侧 exact McNemar；区间为场景聚类风险差的 95% bootstrap CI，p 值未作多重比较校正。", "",
        "| 阶段 / 查询 | n | Native：SR / SPL | GEM：SR / SPL | 配对增 / 损 | p | 风险差 95% CI |",
        "|---|---:|---|---|---|---:|---|"]
    a = verified["A"]
    n = a["constructed"]
    a_score = f'{a["successes"]}/{n} ({100*a["successes"]/n:.1f}%) / {a["mean_spl"]:.3f}' if n else "—"
    lines.append(f'| A / Novel | {n} | {a_score} | 共享 A 历史，不是另一评测臂 | — | — | — |')
    for stage in ("B", "C"):
        for role in ("novel", "revisit"):
            lines.append("| " + " | ".join([f'{stage} / {role.title()}',
                *result_cells(verified["groups"][stage][role])]) + " |")
    lines += ["", "## 2. C：按前序任务展开", "",
        "| 前序 native-B / C 查询 | n | Native：SR / SPL | GEM：SR / SPL | 配对增 / 损 | p | 风险差 95% CI |",
        "|---|---:|---|---|---|---:|---|"]
    for before in ("novel", "revisit"):
        for role in ("novel", "revisit"):
            lines.append("| " + " | ".join([f'{before.title()} → {role.title()}',
                *result_cells(summary["C_by_native_B_role"][before][role])]) + " |")
    counts = verified["C_native_B_source_counts"]
    lines += ["", f'实际 C 参考前缀：Novel-B {counts["novel"]} 条，Revisit-B {counts["revisit"]} 条。'
        "每个前缀派生两种 C 查询；同一来源的派生查询不作为独立场景。", "",
        "## 3. 构造供给与难度", "",
        f'预定 {summary["source_scenes"]} scenes / {summary["source_budget"]} 个 A 源；'
        f'构造出 {n} 个 A 查询，其中 {a["successes"]} 个完成 A。'
        "B 要求同一成功 A 同时可构造两种角色；C 还要求 native-B 成功及两种 C 均可构造。", "",
        "| 查询 | n | 平均初始测地距离 (m) | 距离 × 方向格计数 |",
        "|---|---:|---:|---|"]
    for key in ("A_novel", "B_novel", "B_revisit", "C_novel", "C_revisit"):
        item = summary["supply"][key]
        distance = f'{item["mean_geodesic_m"]:.3f}' if item["queries"] else "—"
        cells = "; ".join(f'{k}: {v}' for k,v in sorted(item.get("distance_direction_cells",{}).items())) or "—"
        lines.append(f'| {key} | {item["queries"]} | {distance} | {cells} |')
    support = summary["supply"]["C_revisit"]["support_sources"]
    lines += ["", "C-Revisit 的实际支持来源：" + ("；".join(f'{k}: {v}' for k,v in sorted(support.items())) or "无查询") + "。",
        "各阶段 SR 不应直接解释为阶段或记忆的因果效果；主比较是同一查询上的 native/GEM。", "",
        "构造损耗（按阶段记录条目计数，B 的两个角色各有一条记录，不等于独立历史数）：", "",
        "| 阶段 | 原始原因 | 条目数 |", "|---|---|---:|"]
    losses = Counter((r["stage"],r["reason"]) for r in summary["attrition"])
    lines += [f'| {stage} | {reason} | {count} |' for (stage,reason),count in sorted(losses.items())]
    if not losses:
        lines.append("| — | 无记录损耗 | 0 |")
    lines += ["", "## 4. 证据路径", "", f'- 汇总：`{summary_path}`',
        f'- 独立复算：`{verification_path}`', f'- Plan SHA-256：`{summary["plan_sha256"]}`',
        f'- Summary SHA-256：`{verified["summary_sha256"]}`',
        f'- 验证任务数：{verified["verified_tasks"]}。', "",
        "本报告不覆盖旧论文表格或旧实验记录；纳入论文前需结合完整分母、效应区间和构造分布解释。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.summary, args.verification)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(report)
