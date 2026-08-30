#!/usr/bin/env python3
"""Fail-closed static audit for the MP3D Table-1 exact-repair wrapper bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit(root: Path) -> dict[str, Any]:
    memnav = root / "MemNavData"
    protocol_path = (
        memnav / "mp3d_table1_controller_exact_repair_protocol_20260829.json"
    )
    navdp_path = memnav / "slurm_hm3d_table1_navdp_pair.sbatch"
    vint_path = memnav / "slurm_hm3d_table1_vint_pair.sbatch"
    helper_path = memnav / "slurm_port_pair.sh"
    submit_path = memnav / "submit_mp3d_table1_controller_exact_repair_remote.sh"
    amendment_path = (
        memnav / "mp3d_table1_navdp_authority_cache_composition_repair_20260829.json"
    )
    repair2_path = (
        memnav / "submit_mp3d_table1_navdp_authority_cache_repair_remote.sh"
    )
    for path in (protocol_path, navdp_path, vint_path, helper_path, submit_path,
                 amendment_path, repair2_path):
        require(path.is_file() and not path.is_symlink(), f"missing physical {path}")

    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("repair_authorized") is True, "repair is not authorized")
    require(protocol.get("method_or_population_changed") is False,
            "scientific contract changed")
    require(protocol.get("frozen_histories") == 42, "history denominator changed")
    require(protocol.get("frozen_scene_clusters") == 25,
            "scene denominator changed")
    require(protocol["navdp"]["failed_array_index"] == 27,
            "NavDP failed rank changed")
    require(protocol["navdp"]["missing_history_indices"] == [29, 30],
            "NavDP repair set changed")
    require(protocol["vint"]["failed_array_index"] == 24,
            "ViNT failed index changed")
    require(protocol["vint"]["missing_history_indices"] == [24],
            "ViNT repair set changed")
    incident = protocol["outcome_visibility_incident"]
    require(incident.get("occurred") is True, "visibility incident omitted")
    require(incident.get("repair_selection_influenced") is False,
            "repair selection was outcome-influenced")

    navdp = navdp_path.read_text()
    require("EXACT_REPAIR" in navdp, "NavDP exact-repair gate missing")
    require("FORMAL_INDICES_SPEC" in navdp, "NavDP history override missing")
    require("FORMAL_INDICES_OVERRIDE=${FORMAL_INDICES_SPEC//:/ }" in navdp,
            "NavDP history override parser changed")
    require("WRAPPER_RECEIPT" in navdp, "NavDP wrapper provenance missing")

    vint = vint_path.read_text()
    require("claim_slurm_tcp_port_block mp3d_table1_vint 6 12000 8000" in vint,
            "ViNT six-port reservation missing")
    require("trap cleanup_port_block EXIT INT TERM" in vint,
            "ViNT lifetime port lock missing")
    require("port_slot=$(( $$ % 6000 ))" not in vint,
            "racy ViNT PID allocator remains")
    require("WRAPPER_RECEIPT" in vint, "ViNT wrapper provenance missing")

    helper = helper_path.read_text()
    require("flock -n" in helper and "ss -H -ltn" in helper,
            "port helper lacks lock/listener checks")
    submit = submit_path.read_text()
    require("nav != [29,30] or vint != [24]" in submit,
            "remote missing-receipt gate changed")
    require("FORMAL_INDICES_SPEC=29:30" in submit,
            "remote NavDP exact set changed")
    require("--array=24" in submit, "remote ViNT exact set changed")
    require("partial_outcome_visibility_incident':True" in submit,
            "remote submission disclosure missing")
    require("repair_selection_influenced_by_incident':False" in submit,
            "remote submission outcome-isolation receipt missing")
    amendment = json.loads(amendment_path.read_text())
    require(amendment.get("attempt") == 2, "composition repair attempt changed")
    require(amendment.get("frozen_history_indices") == [29, 30],
            "composition repair set changed")
    require(amendment.get("method_or_population_changed") is False,
            "composition repair changed scientific contract")
    require(amendment["outcome_visibility"].get(
        "navigation_success_or_distance_read") is False,
        "composition diagnosis read navigation outcome")
    repair2 = repair2_path.read_text()
    require("SERVER_SOURCE_ROOT=${TASK}" in repair2,
            "repair2 does not use the composed task source")
    require("FORMAL_INDICES_SPEC=29:30" in repair2,
            "repair2 exact set changed")
    require("navigation_success_or_distance_read':False" in repair2,
            "repair2 disclosure missing")
    require('assert_job "${VINT_VERIFY}" COMPLETED 0:0' in repair2,
            "repair2 does not require a completed retained ViNT verifier")
    require("EXPECTED_VINT_VERIFICATION_SHA=${VINT_VERIFICATION_SHA}" in repair2,
            "repair2 does not hash-pin the retained ViNT verifier")
    require("--dependency=afterok:${verify}:${VINT_VERIFY}" not in repair2,
            "repair2 reattaches an already-completed job to the seal dependency")
    require("--dependency=afterok:${verify} \\" in repair2,
            "repair2 seal does not depend on the new NavDP verifier")
    require("replacement_seal_depends_on_navdp_verify_only':True" in repair2,
            "repair2 scheduler correction receipt missing")
    return {
        "verified": True,
        "frozen_histories": 42,
        "frozen_scene_clusters": 25,
        "navdp_exact_histories": [29, 30],
        "vint_exact_histories": [24],
        "method_or_population_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
