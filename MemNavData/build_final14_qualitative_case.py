#!/usr/bin/env python3
"""Build the frozen Final14 qualitative motivation case.

The figure is a post-hoc visualization of one already consumed Final14
history.  It does not create a new evaluation row and it does not use the
case to select a threshold.  The input directory is expected to contain the
three paired controller plans, their metric CSVs, and the RGB files copied
from the immutable evaluation bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


ARMS = ("mono_native", "mono_raw_fixed", "mono_cec")
ROLES = ("novel", "revisit")
COLORS = {
    "mono_native": "#6F7782",
    "mono_raw_fixed": "#D9822B",
    "mono_cec": "#356FA3",
}
LABELS = {
    "mono_native": "Native",
    "mono_raw_fixed": "Raw memory",
    "mono_cec": "CEC",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_metric(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2, f"expected two role rows in {path}")
    result = {row["analysis_role"]: row for row in rows}
    require(set(result) == set(ROLES), f"role rows changed in {path}")
    return result


def load_inputs(root: Path):
    plans = {
        (arm, role): read_json(root / f"{arm}_{role}_plans.json")
        for arm in ARMS for role in ROLES
    }
    metrics = {
        arm: read_metric(root / f"{arm}_metric.csv") for arm in ARMS
    }
    role_pairs = read_json(root / "role_pairs.json")

    expected = {
        ("mono_native", "novel"): 1,
        ("mono_raw_fixed", "novel"): 0,
        ("mono_cec", "novel"): 1,
        ("mono_native", "revisit"): 0,
        ("mono_raw_fixed", "revisit"): 1,
        ("mono_cec", "revisit"): 1,
    }
    for key, value in expected.items():
        arm, role = key
        require(int(metrics[arm][role]["reached"]) == value,
                f"frozen outcome changed for {arm}/{role}")

    novel_cec = plans[("mono_cec", "novel")]
    revisit_cec = plans[("mono_cec", "revisit")]
    require(not any(bool(row.get("revisit_adapter_takeover"))
                    for row in novel_cec["query_leg"]),
            "CEC unexpectedly takes over the Novel query")
    require(any(bool(row.get("revisit_adapter_takeover"))
                for row in revisit_cec["query_leg"]),
            "CEC never takes over the Revisit query")
    require(
        novel_cec["memory_traces"]["query"]
        == plans[("mono_native", "novel")]["memory_traces"]["query"],
        "CEC rejection is not an exact native trajectory fallback",
    )

    first_novel = novel_cec["query_leg"][0]
    first_revisit = revisit_cec["query_leg"][0]
    require(first_novel["certified_relocalization_accepted"] is False,
            "Novel certificate no longer rejects")
    require(first_revisit["certified_relocalization_accepted"] is True,
            "Revisit certificate no longer accepts")
    require(first_novel["router_selected_anchor"] == 151,
            "frozen Novel witness anchor changed")
    require(first_revisit["router_selected_anchor"] == 69,
            "frozen Revisit witness anchor changed")

    queries = {
        item["analysis_role"]: item
        for item in role_pairs["pairs"][0]["queries"]
    }
    require(set(queries) == set(ROLES), "role-pair query set changed")
    return plans, metrics, queries, first_novel, first_revisit


def run_matches(reference: Path, query: Path, lightglue_root: Path):
    sys.path.insert(0, str(lightglue_root.resolve()))
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher

    matcher = LightGluePointMatcher(
        lightglue_root, dependency_root=None, device="cuda:0",
        max_keypoints=2048)
    matched = matcher.match_paths(
        reference, query, target_height=518, target_width=518)
    points0 = np.asarray(matched["reference_raw_points"], dtype=np.float32)
    points1 = np.asarray(matched["query_raw_points"], dtype=np.float32)
    cv2.setRNGSeed(0)
    _fundamental, mask = cv2.findFundamentalMat(
        points0, points1, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
    require(mask is not None, f"Fundamental-MAGSAC failed for {reference}")
    inliers = np.asarray(mask).reshape(-1).astype(bool)
    require(len(inliers) == len(points0), "Fundamental mask size changed")
    return points0, points1, inliers


def draw_match_inset(ax, reference: Path, query: Path, match,
                     *, title: str, subtitle: str, seed: int) -> None:
    ref = np.asarray(Image.open(reference).convert("RGB"))
    goal = np.asarray(Image.open(query).convert("RGB"))
    height = max(ref.shape[0], goal.shape[0])
    width = ref.shape[1] + goal.shape[1]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[:ref.shape[0], :ref.shape[1]] = ref
    canvas[:goal.shape[0], ref.shape[1]:] = goal
    ax.imshow(canvas)
    p0, p1, inliers = match
    indices = np.flatnonzero(inliers)
    rng = np.random.default_rng(seed)
    if len(indices) > 22:
        indices = np.sort(rng.choice(indices, 22, replace=False))
    for index in indices:
        x0, y0 = p0[index]
        x1, y1 = p1[index]
        ax.plot([x0, x1 + ref.shape[1]], [y0, y1],
                color="#3F8F63", linewidth=0.55, alpha=0.78)
        ax.scatter([x0, x1 + ref.shape[1]], [y0, y1], s=3.2,
                   color="#3F8F63", edgecolors="white", linewidths=0.2)
    ax.axvline(ref.shape[1] - 0.5, color="white", linewidth=1.0)
    ax.text(0.012, 0.965, "historical candidate", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color="#24272C",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                  "pad": 1.4})
    ax.text(0.988, 0.965, "goal image", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="#24272C",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                  "pad": 1.4})
    ax.set_title(title, loc="left", fontsize=12.5, weight="semibold", pad=3)
    ax.text(0.0, -0.035, subtitle, transform=ax.transAxes, va="top",
            fontsize=9.5, color="#4D535B")
    ax.axis("off")


def trajectory(plan: dict) -> np.ndarray:
    rows = plan["memory_traces"]["query"]
    return np.asarray([[float(row["x"]), float(row["z"])] for row in rows])


def draw_trajectory_panel(ax, role: str, plans: dict,
                          goal: list[float]) -> None:
    traces = {arm: trajectory(plans[(arm, role)]) for arm in ARMS}
    for arm in ARMS:
        trace = traces[arm]
        ax.plot(trace[:, 0], trace[:, 1], color=COLORS[arm], linewidth=1.7,
                alpha=0.95, label=LABELS[arm], zorder=2)
    start = traces["mono_native"][0]
    ax.scatter(*start, marker="o", s=30, color="#24272C", zorder=5)
    ax.scatter(float(goal[0]), float(goal[2]), marker="*", s=92,
               color="#3F8F63", edgecolor="white", linewidth=0.7, zorder=6)
    ax.text(start[0], start[1], "  start", fontsize=9.0, va="center")
    ax.text(float(goal[0]), float(goal[2]), "  goal", fontsize=9.0,
            color="#2E7350", va="center")

    title = ("Unsupported Novel: raw memory interferes; CEC abstains"
             if role == "novel" else
             "Supported Revisit: verified memory restores navigation")
    ax.set_title(title, loc="left", fontsize=13.0, weight="semibold",
                 y=1.035, pad=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CDD1D6")
        spine.set_linewidth(0.7)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    bar = 1.0
    x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.07 * (ymax - ymin)
    ax.plot([x0, x0 + bar], [y0, y0], color="#24272C", linewidth=1.5)
    ax.text(x0 + bar / 2, y0 + 0.02 * (ymax - ymin), "1 m",
            ha="center", va="bottom", fontsize=9.0)


def image_panel(ax, path: Path, title: str) -> None:
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_title(title, loc="left", fontsize=11.5, weight="semibold", pad=3)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lightglue-root", type=Path,
                        default=Path(".diagnostics/dependencies/LightGlue"))
    args = parser.parse_args()
    root = args.assets.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    plans, metrics, queries, novel_row, revisit_row = load_inputs(root)
    novel_match = run_matches(
        root / "novel_candidate.jpg", root / "novel_goal.jpg",
        args.lightglue_root)
    revisit_match = run_matches(
        root / "cec_anchor.jpg", root / "revisit_goal.jpg",
        args.lightglue_root)

    novel_pnp = novel_row["certified_relocalization_pnp"]
    revisit_pnp = revisit_row["certified_relocalization_pnp"]
    # Feature extraction can move by one or two boundary matches across CUDA
    # architectures.  The plotted lines are therefore a fresh visualization
    # of the byte-identical RGB pair, while every numeric annotation below is
    # read from the frozen formal receipt.  Fail closed on any material drift.
    frozen_fundamental = {
        "novel": int(novel_row["router_candidate_trials"][0][
            "fundamental_inliers"]),
        "revisit": int(revisit_row["router_candidate_trials"][0][
            "fundamental_inliers"]),
    }
    visual_fundamental = {
        "novel": int(np.sum(novel_match[2])),
        "revisit": int(np.sum(revisit_match[2])),
    }
    for role in ROLES:
        require(abs(visual_fundamental[role] - frozen_fundamental[role]) <= 2,
                f"{role} visualization materially differs from frozen evidence")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(13.6, 4.25), facecolor="white")
    grid = fig.add_gridspec(
        2, 6, height_ratios=(0.72, 1.18),
        width_ratios=(0.75, 1.1, 1.1, 0.75, 1.1, 1.1),
        hspace=0.30, wspace=0.22)

    image_panel(fig.add_subplot(grid[0, 0]), root / "current.jpg",
                "Shared current RGB")
    draw_match_inset(
        fig.add_subplot(grid[0, 1:3]), root / "novel_candidate.jpg",
        root / "novel_goal.jpg", novel_match,
        title="Novel query: plausible local match, insufficient witness",
        subtitle=(f"{int(novel_pnp['inliers'])} PnP inliers · "
                  f"reference coverage {100*novel_pnp['reference_inlier_coverage']:.1f}% < 5% · reject → native"),
        seed=17)
    image_panel(fig.add_subplot(grid[0, 3]), root / "current.jpg",
                "Same current RGB")
    draw_match_inset(
        fig.add_subplot(grid[0, 4:6]), root / "cec_anchor.jpg",
        root / "revisit_goal.jpg", revisit_match,
        title="Revisit query: distributed evidence supports a pose witness",
        subtitle=(f"{int(revisit_pnp['inliers'])} PnP inliers · coverage "
                  f"{100*revisit_pnp['query_inlier_coverage']:.1f}% / "
                  f"{100*revisit_pnp['reference_inlier_coverage']:.1f}% · "
                  f"RMSE {revisit_pnp['reprojection_rmse_px']:.2f} px · accept → bearing"),
        seed=23)

    draw_trajectory_panel(
        fig.add_subplot(grid[1, :3]), "novel", plans,
        queries["novel"]["floor_position"])
    draw_trajectory_panel(
        fig.add_subplot(grid[1, 3:]), "revisit", plans,
        queries["revisit"]["floor_position"])

    legend = [
        Line2D([0], [0], color=COLORS[arm], lw=2.0, label=LABELS[arm])
        for arm in ARMS
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=11.0)
    fig.subplots_adjust(left=0.018, right=0.988, top=0.955, bottom=0.105)

    png = output / "final14_qualitative_failure_case.png"
    pdf = output / "final14_qualitative_failure_case.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    receipt = {
        "schema_version": "final14_qualitative_case_v1_20260830",
        "scope": "posthoc_visualization_of_frozen_final14_case",
        "case_selection": (
            "representative paired case; not used for threshold or model selection"),
        "runtime_role_visible": False,
        "scientific_rows_added": 0,
        "frozen_outcomes": {
            role: {LABELS[arm]: int(metrics[arm][role]["reached"])
                   for arm in ARMS}
            for role in ROLES
        },
        "certificate": {
            "novel": {
                "accepted": False,
                "reason": novel_row["certified_relocalization_reason"],
                "pnp_inliers": int(novel_pnp["inliers"]),
                "reference_coverage": float(
                    novel_pnp["reference_inlier_coverage"]),
            },
            "revisit": {
                "accepted": True,
                "reason": revisit_row["certified_relocalization_reason"],
                "pnp_inliers": int(revisit_pnp["inliers"]),
                "query_coverage": float(revisit_pnp["query_inlier_coverage"]),
                "reference_coverage": float(
                    revisit_pnp["reference_inlier_coverage"]),
                "reprojection_rmse_px": float(
                    revisit_pnp["reprojection_rmse_px"]),
            },
        },
        "match_visualization": {
            "contract": (
                "byte-identical RGB rerun; numeric figure annotations come "
                "from frozen formal PnP receipts"),
            "frozen_fundamental_inliers": frozen_fundamental,
            "visualization_fundamental_inliers": visual_fundamental,
        },
        "source_files": {
            path.name: sha256(path)
            for path in sorted(root.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        },
        "outputs": {png.name: sha256(png), pdf.name: sha256(pdf)},
    }
    receipt_path = output / "final14_qualitative_failure_case_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "png": str(png), "pdf": str(pdf),
        "receipt": str(receipt_path), "verified": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
