#!/usr/bin/env python3
"""Frozen consumed four-history, three-arm local X-NavDP stack diagnostic."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.run_cec_stream_depth_closed_loop import BENCH, MEM_PY
from MemNavData.run_habitat_bullet_query_local import BULLET_PY
from MemNavData.habitat_xnavdp_tracking import CONFIG, SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primitive", type=Path, required=True)
    parser.add_argument("--rgb-pair", action="store_true",
                        help="Isolate legacy BGR versus corrected RGB in the same X actor")
    parser.add_argument("--rgb-audit", type=Path)
    args = parser.parse_args()
    primitive = args.primitive.resolve() / "summary.json"
    proof = json.loads(primitive.read_text())
    assert proof["passed"] and proof["config"] == CONFIG and proof["source_sha256"] == sha(SOURCE)
    if args.rgb_pair:
        assert args.rgb_audit is not None
        color_proof = json.loads(args.rgb_audit.read_text())
        assert color_proof["passed"] and len(color_proof["cases"]) == 5
        for name, digest in color_proof["source_sha256"].items():
            assert sha(Path(name)) == digest
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    histories = json.loads((BENCH / "manifest.json").read_text())["episodes"]
    assert [h["scene"] for h in histories] == ["gxdoqLR6rwA", "pLe4wQe7qrG", "yqstnuAEVhm", "mJXqzFtmKg4"]
    with (out / "logs/unit_preflight.log").open("x") as stream:
        subprocess.run([MEM_PY, "-m", "unittest", "MemNavData.test_habitat_xnavdp_tracking",
                        "MemNavData.test_habitat_bullet_alignment",
                        "MemNavData.test_xnavdp_rgb_contract",
                        "MemNavData.test_xnavdp_revisit_contract", "-v"],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    runner = ROOT / "MemNavData/run_habitat_bullet_query_local.py"
    plan = [{"index": i, "scene": h["scene"], "episode": h["episode"],
             "history_sha256": h["online_a_trace_sha256"]} for i, h in enumerate(histories)]
    dump(out / "manifest.json", dict(histories=plan,
         arms=(["cec_x_legacy__xmpc", "cec_x__xmpc"] if args.rgb_pair
               else ["cec_aligned__mpc", "cec__xmpc", "cec_x__xmpc"]),
         rgb_pair=args.rgb_pair,
         rgb_audit_sha256=sha(args.rgb_audit) if args.rgb_pair else None,
         balanced_order="legacy first on even histories, corrected first on odd" if args.rgb_pair else None,
         source_manifest_sha256=sha(BENCH / "manifest.json"), runner_sha256=sha(runner),
         protocol_sha256=sha(ROOT / ("MemNavData/HABITAT_XNAVDP_RGB_REPAIR_PROTOCOL_20260908.md"
                             if args.rgb_pair else "MemNavData/HABITAT_XNAVDP_STACK_PROTOCOL_20260908.md")),
         primitive_sha256=sha(primitive), scope="consumed diagnostic, no formal SR or new training"))
    results = []
    for item in plan:
        run = out / f"history_{item['index']:02d}_{item['scene']}"
        dump(out / "progress.json", dict(active=item, active_run=str(run), completed=results, stage="navigation"))
        print(f"START {item['index']} {item['scene']}: " +
              ("paired legacy/corrected X RGB" if args.rgb_pair else "three X/base stack arms"), flush=True)
        started = time.monotonic()
        with (out / f"logs/history_{item['index']:02d}.log").open("x") as stream:
            code = subprocess.run([str(BULLET_PY), "-u", str(runner), "local", "--out", str(run),
                                   "--history-index", str(item["index"]), "--max-steps", "600",
                                   "--xnavdp-rgb-pair" if args.rgb_pair else "--xnavdp-pair",
                                   "--continue-after-invalid-physics"],
                                  cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        row = dict(**item, run=str(run), exit_code=code, elapsed_s=time.monotonic()-started)
        for key in ("summary", "failure"):
            if (run / f"{key}.json").exists():
                row[key] = json.loads((run / f"{key}.json").read_text())
        results.append(row)
        dump(out / "partial_results.json", results)
        print(f"DONE {item['index']} {item['scene']}: exit={code}", flush=True)
        if code not in (0, 2):
            dump(out / "failure.json", dict(history=item, exit_code=code,
                 message="Infrastructure/interface failure; do not count as policy outcome"))
            return code
    dump(out / "summary.json", dict(all_histories_attempted=True, results=results, formal_SR_claim=False))
    dump(out / "progress.json", dict(stage="navigation_complete", completed=results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
