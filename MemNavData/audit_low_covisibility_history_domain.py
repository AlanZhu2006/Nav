#!/usr/bin/env python3
"""Read archived covisibility curves without rerendering or changing labels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    evidence = [json.loads(line.split(" ", 1)[1]) for line in args.evidence.read_text().splitlines()
                if line.startswith("EVIDENCE_ROW ")]
    by_history = {}
    known = {}
    for row in evidence:
        by_history[row["history"]] = row
        known[row["history"], row["group"]] = row
    records = []
    source_hashes = {}
    for history, source in sorted(by_history.items()):
        path = Path(source["provenance"]["construction"])
        data = path.read_bytes()
        source_hashes[str(path)] = hashlib.sha256(data).hexdigest()
        construction = json.loads(data)
        count = construction["online_a_steps"]
        for q in construction["queries"]:
            curve = q["covis_curve"]
            if len(curve) != count or abs(max(curve[39:])-q["q_eligible"]) > 1e-10:
                raise ValueError("Archived annotation convention changed")
            q8 = max(curve[8:])
            record = {"history": history, "scene": construction["scene"], "query_id": q["query_id"],
                "q_annotation_frame39": q["q_eligible"], "q_cec_frame8": q8,
                "q_all": q["q_all"], "argmax_cec_frame8": 8+curve[8:].index(q8),
                "q_raw_first_decision_frame39_to_n_minus32": max(curve[39:count-32+1]),
                "history_frames": count, "curve": curve}
            if (history, q["query_id"]) in known:
                row = known[history, q["query_id"]]
                raw_anchor = row["arms"]["raw_fixed"]["plan"]["anchor"]
                cec_anchor = row["arms"]["cec"]["plan"]["router_selected_anchor"]
                record.update(task=row["task"], raw_anchor=raw_anchor, cec_anchor=cec_anchor,
                              q_raw_anchor=curve[raw_anchor], q_cec_anchor=curve[cec_anchor])
            records.append(record)
    report = {"scope": "posthoc history-domain audit; no relabeling, no inference, no navigation",
        "annotation_floor": 39, "cec_runtime_floor": 8, "raw_first_decision_exclude_recent": 32,
        "source_hashes": source_hashes, "queries": records}
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"path": str(args.out), "queries": len(records),
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
