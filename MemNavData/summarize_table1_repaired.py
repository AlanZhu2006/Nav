"""Six fixed controller/dataset pairs, role-separated SR/SPL and interference."""
import argparse
from pathlib import Path
import numpy as np

from table1_repaired_eval import dump, load, sha
from summarize_covisibility_repaired import exact_mcnemar, holm


def statistics(rows):
    scenes = sorted({r["scene"] for r in rows})
    effects = [r["cec"]["reached"] - r["native"]["reached"] for r in rows]
    totals = np.array([sum(e for r,e in zip(rows,effects) if r["scene"] == s) for s in scenes])
    counts = np.array([sum(r["scene"] == s for r in rows) for s in scenes])
    sample = np.random.default_rng(20260910).integers(0,len(scenes),(20000,len(scenes)))
    bootstrap = totals[sample].sum(1)/counts[sample].sum(1)
    gain, loss = effects.count(1), effects.count(-1)
    return dict(queries=len(rows),scenes=len(scenes),arms={a:dict(
        successes=sum(r[a]["reached"] for r in rows),sr=float(np.mean([r[a]["reached"] for r in rows])),
        spl=float(np.mean([r[a]["spl"] for r in rows]))) for a in ("native","cec")},
        gain=gain,loss=loss,risk_difference=float(np.mean(effects)),exact_mcnemar_p=exact_mcnemar(gain,loss),
        scene_cluster_bootstrap_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        cec_takeover_queries=sum(r["pair"]["cec_takeover"] for r in rows),
        exact_native_queries=sum(r["pair"]["no_takeover_exact_native"] for r in rows))


def summarize(plan_path, root, output):
    if output.exists():
        raise FileExistsError(output)
    plan = load(plan_path)
    rows, missing = [], []
    for cell in plan["cells"]:
        folder = root / f'task_{cell["index"]:03d}'
        path = folder / "archive_receipt.json"
        archive = load(path) if path.exists() else {}
        if not archive.get("completed"):
            missing.append(dict(index=cell["index"],reason="missing/incomplete archive",exit_code=archive.get("exit_code")))
            continue
        v = load(folder / "independent_verification.json")
        assert v == archive["verification"] and v["verified"] and v["cell"] == cell
        assert archive["all_member_hashes_readback_verified"] and sha(archive["archive"]) == archive["archive_sha256"]
        assert len(v["records"]) == 4 and len(v["pairs"]) == 2
        for pair in v["pairs"]:
            arms = {r["arm"]:r for r in v["records"] if r["role"] == pair["role"]}
            assert set(arms) == {"native","cec"}
            assert all(arms[a]["reached"] == pair[a] for a in arms)
            rows.append(dict(cell,role=pair["role"],pair=pair,**arms))
    if missing:
        dump(output,dict(complete=False,expected_cells=len(plan["cells"]),verified_cells=len(rows)//2,
            missing_or_failed_cells=missing,note="Infrastructure loss is not navigation failure; no complete-population SR."))
        raise RuntimeError(f"Incomplete population: {len(missing)} cells")
    groups = {}
    for dataset in ("hm3d","mp3d"):
        for controller in ("navdp","vint","nomad"):
            subset = [r for r in rows if r["dataset"] == dataset and r["controller"] == controller]
            groups[f"{dataset}_{controller}"] = {role:statistics([r for r in subset if role == "all" or r["role"] == role])
                                               for role in ("novel","revisit","all")}
    adjusted = holm([g["revisit"]["exact_mcnemar_p"] for g in groups.values()])
    for group,p in zip(groups.values(),adjusted):
        group["revisit"]["holm_six_groups_p"] = p
    dump(output,dict(complete=True,plan_sha256=sha(plan_path),cells=len(plan["cells"]),rollouts=2*len(rows),
        groups=groups,paired_rows=rows,bootstrap_replicates=20000,bootstrap_seed=20260910,
        scope="Corrected query-time portability on existing Table-I histories; not fresh full-system A or an official topomap benchmark."))
    print("SEALED 210 complete paired cells / 840 rollouts",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a = parser.parse_args()
    summarize(a.plan,a.root,a.output)
