#!/usr/bin/env python3
"""Package consumed RGB histories for the fixed coverage-ablation diagnostic."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", choices=("first_two", "missing_coverage_pnp"),
                        default="first_two")
    args = parser.parse_args()
    rows = [json.loads(line.split(" ", 1)[1]) for line in args.evidence.read_text().splitlines()
            if line.startswith("EVIDENCE_ROW ")]
    if args.selection == "missing_coverage_pnp":
        statuses = {"precheck_fundamental_query_hull_coverage",
                    "precheck_fundamental_reference_hull_coverage"}
        rows = [r for r in rows if r["arms"]["cec"]["plan"][
            "certified_relocalization_pnp"]["status"] in statuses]
        if sorted(r["task"] for r in rows) != [7, 24, 37, 69, 71, 129]:
            raise ValueError("Frozen missing-PnP population changed")
        selected_histories = sorted({r["history"] for r in rows})
        selection = "all six original area-precheck rejections; no outcome selection"
    else:
        selected_histories = sorted({r["history"] for r in rows})[:2]
        selection = "first two history indices, independent of outcomes"
    files, histories, evaluation = [], [], {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "x:gz", compresslevel=1) as out:
        def add_bytes(name, data, expected=None):
            if expected is not None and digest(data) != expected:
                raise ValueError(f"Input changed: {name}")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            out.addfile(info, io.BytesIO(data))
            files.append({"path": name, "sha256": digest(data), "bytes": len(data)})

        for history in selected_histories:
            queries = sorted([r for r in rows if r["history"] == history], key=lambda r: r["group"])
            row = queries[0]
            source = Path(row["provenance"]["online_a_episode"])
            receipt = json.loads((source / "receipt.json").read_text())
            construction = json.loads(Path(row["provenance"]["construction"]).read_text())
            prefix = f"history_{history:02d}"
            n = len(receipt["rgb_frame_hashes"])
            for i, expected in enumerate(receipt["rgb_frame_hashes"]):
                add_bytes(f"{prefix}/rgb/{i}.jpg", (source / f"rgb/{i:06d}.jpg").read_bytes(), expected)
            if row["arms"]["cec"]["plan"]["frame_idx"] != n:
                raise ValueError("Expected exactly one query-start observation after A prefix")
            root = Path(row["provenance"]["construction"]).parents[2] / "evaluation_f284c23d7a98b0c2"
            archived = json.loads((root / f"task_{row['task']:03d}/archive_receipt.json").read_text())
            current_sha = row["arms"]["cec"]["image_sha256"]
            current_member = None
            with tarfile.open(archived["archive"], "r|gz") as stream:
                for member in stream:
                    if member.name == "task/artifact_index.json":
                        entries = json.load(stream.extractfile(member))["files"]
                        candidates = [e for e in entries if e["sha256"] == current_sha and e["path"].endswith(".jpg")]
                        if not candidates:
                            raise ValueError("Query-start RGB not retained in archive")
                        current_member = "task/" + candidates[0]["path"]
                    if member.name == current_member:
                        add_bytes(f"{prefix}/rgb/{n}.jpg", stream.extractfile(member).read(), current_sha)
                        break
                else:
                    raise ValueError("Current RGB missing from archive")
            trace = json.loads((source / "online_a_trace.json").read_text())
            case = {"history": history, "scene": row["scene"], "rgb": f"{prefix}/rgb",
                    "frames": n+1, "seed": int(trace["episode_seed"]), "episode_len": n+600,
                    "camera_height_m": receipt["camera_height_m"],
                    "camera_intrinsic": construction["annotation_intrinsic"], "queries": []}
            for query in queries:
                name = f"{prefix}/goals/{query['group']}.jpg"
                add_bytes(name, Path(query["provenance"]["goal_rgb"]).read_bytes(),
                          query["provenance"]["goal_rgb_sha256"])
                trials = query["arms"]["cec"]["plan"]["router_candidate_trials"]
                shortlist = [{"anchor": c["anchor"], "score": c["dino_cosine"]}
                             for c in sorted(trials, key=lambda c: c["dino_rank"])]
                case["queries"].append({"task": query["task"], "goal": name, "candidates": shortlist})
                evaluation[str(query["task"])] = {
                    "group": query["group"], "q_eligible": query["q_eligible"],
                    "state": query["arms"]["cec"]["state"],
                    "goal_floor_position": query["goal_floor_position"]}
            histories.append(case)
        add_bytes("runtime_inputs.json", json.dumps({"histories": histories,
                  "selection": selection,
                  "scope": "fixed-proposal RGB replay; no navigation SR"}).encode())
        add_bytes("evaluation_only.json", json.dumps(evaluation).encode())
        add_bytes("input_files.json", json.dumps(files).encode())
    data = args.out.read_bytes()
    print(json.dumps({"path": str(args.out), "histories": selected_histories,
          "queries": sum(len(h["queries"]) for h in histories),
          "bytes": len(data), "sha256": digest(data)}, indent=2))


if __name__ == "__main__":
    main()
