#!/usr/bin/env python3
"""CPU-only X MPC preflight before consumed-history navigation."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_xnavdp_tracking import ACADOS, CONFIG, SOURCE, XTracker
from MemNavData.habitat_executor_audit import dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["X_NAVDP_MPC_CODEGEN_DIR"] = str(out / "codegen")
    tracker = XTracker()
    cases = []
    for index, sign in enumerate((0, 1, -1)):
        # Habitat yaw zero is forward -Z; +X in MPC is local forward.
        world = np.column_stack((np.zeros(24), -sign*np.linspace(.08, 2, 24)))
        rows = [tracker.command(index, np.zeros(3), 0., world) for _ in range(8)]
        controls = np.array([[row[0], row[1]] for row in rows])
        assert [r[3]["consumed_index"] for r in rows] == list(range(8))
        assert rows[0][2] is not None and all(r[2] is None for r in rows[1:])
        assert abs(controls[:, 1]).max() < 1e-3
        if sign:
            assert np.min(controls[:, 0]*sign) > .01
        else:
            assert abs(controls).max() < 1e-4
        cases.append(dict(sign=sign, controls=controls.tolist(),
                          solve_ms=1000*rows[0][2], receipt=rows[0][3]))
    dump(out / "summary.json", dict(passed=True, cases=cases, config=CONFIG,
         source_sha256=sha(SOURCE), acados_source=str(ACADOS),
         scope="CPU solver/coordinate contract only; no neural policy or SR"))
    print(json.dumps({"passed": True, "cases": len(cases)}))


if __name__ == "__main__":
    main()
