#!/usr/bin/env python3
"""Independently verify the consumed dense-query mechanism-gate summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from MemNavData.analyze_hm3d_longrange_dense_query_gate import (
    build_summary,
    require,
    sha256_file,
)


SCHEMA = "hm3d_longrange_dense_live_query_gate_verification_v2_20260903"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run_root / "result" / "summary.json"
    sidecar = summary_path.with_name(summary_path.name + ".sha256")
    require(summary_path.is_file() and sidecar.is_file(),
            "dense-query summary or sidecar is missing")
    words = sidecar.read_text().strip().split()
    require(len(words) == 2 and words[0] == sha256_file(summary_path)
            and words[1] == summary_path.name,
            "dense-query summary sidecar changed")
    reported = json.loads(summary_path.read_text())
    recounted = build_summary(args.run_root, args.protocol)
    require(reported == recounted,
            "independent dense-query recount disagrees with summary")
    verification = {
        "schema_version": SCHEMA,
        "verified": True,
        "summary_sha256": sha256_file(summary_path),
        "protocol_sha256": sha256_file(args.protocol),
        "histories": recounted["histories"],
        "geometry_stream_failures": recounted[
            "geometry_stream_failures"],
        "mechanism_gate_passed": recounted["mechanism_gate_passed"],
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
    }
    path = args.run_root / "result" / "independent_verification.json"
    require(not path.exists(), f"verification already exists: {path}")
    encoded = (json.dumps(
        verification, indent=2, sort_keys=True, allow_nan=False)
        + "\n").encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n")
    print(json.dumps(verification, sort_keys=True))


if __name__ == "__main__":
    main()
