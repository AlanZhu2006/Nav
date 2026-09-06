#!/usr/bin/env python3
"""Run one sealed HM3D Table-I history under the four authority-spectrum arms.

The mature full-monocular history runner owns replay, role-hiding, controller
depth, exact-fallback, and receipt checks.  This narrow entry point installs the
pre-registered four-arm contract and then adds a matched-proposal audit between
the finite-PnP and strict-certificate arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from MemNavData import run_hm3d_fullmono_query_history as base
from MemNavData.final14_authority_ablation import AUTHORITY_POLICY
from MemNavData.run_final14_authority_ablation_episode import (
    audit_initial_proposal_pair,
)
from MemNavData.run_final14_mono_factorial_episode import load_payloads, load_rows
from MemNavData.hm3d_table1_authority_spectrum import (
    ARMS,
    DEPTH_SOURCE,
    EVALUATOR_ARM,
    HYBRID_ROUTE,
    REVISIT_ADAPTER,
    audit_query_arm,
    selected_arm_order,
)


SCHEMA = "hm3d_table1_authority_spectrum_history_v1_20260904"


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    return parser.parse_known_args(argv)[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_contract() -> None:
    base.ARMS = ARMS
    base.DEPTH_SOURCE = DEPTH_SOURCE
    base.EVALUATOR_ARM = EVALUATOR_ARM
    base.HYBRID_ROUTE = HYBRID_ROUTE
    base.REVISIT_ADAPTER = REVISIT_ADAPTER
    base.audit_query_arm = audit_query_arm
    base.selected_arm_order = selected_arm_order
    base.SCHEMAS["goal_a"] = SCHEMA


def _append_proposal_receipt(args: argparse.Namespace) -> None:
    manifest = json.loads((args.bench_root / "manifest.json").read_text())
    item = manifest["episodes"][args.history_index]
    label = f'{args.history_index:03d}_{item["scene"]}_{item["episode"]}'
    root = args.run_root / "evaluation" / "natural_direction" / label

    payloads = {}
    for arm in ("mono_unthresholded_witness", "mono_cec"):
        rows = load_rows(root / arm)
        payloads[arm] = load_payloads(
            root / arm, str(item["episode"]), rows
        )
    proposal_audit = {
        role: audit_initial_proposal_pair(
            payloads["mono_cec"][role],
            payloads["mono_unthresholded_witness"][role],
            label=f"{label}/{role}",
        )
        for role in ("novel", "revisit")
    }

    completion_path = root / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion.update({
        "schema_version": SCHEMA,
        "experiment_scope": (
            "sealed_hm3d_table1_retrospective_authority_spectrum"
        ),
        "fresh_confirmation": False,
        "outcomes_known_before_ablation_design": ["mono_native", "mono_cec"],
        "raw_and_witness_outcomes_known_before_submission": False,
        "initial_proposal_equality": True,
        "initial_proposal_audit": proposal_audit,
        "authority_policies": AUTHORITY_POLICY,
        "causal_ladder": [
            "native_no_memory",
            "raw_dino_top1_fixed_bearing",
            "proposal_matched_finite_pnp_witness",
            "strict_operational_certificate",
        ],
    })
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    completion_path.write_bytes(encoded)
    (root / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n"
    )

    contract_path = root / "episode_contract.json"
    contract = json.loads(contract_path.read_text())
    contract.update({
        "schema_version": SCHEMA,
        "experiment_scope": (
            "sealed_hm3d_table1_retrospective_authority_spectrum"
        ),
        "fresh_confirmation": False,
        "sole_matched_proposal_intervention": (
            "finite_pnp_authority_vs_strict_certificate"
        ),
    })
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = _arguments(sys.argv[1:])
    _install_contract()
    base.main()
    _append_proposal_receipt(args)


if __name__ == "__main__":
    main()
