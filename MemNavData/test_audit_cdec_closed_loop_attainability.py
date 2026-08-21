import copy
import hashlib
from pathlib import Path

import pytest

from MemNavData import audit_cdec_closed_loop_attainability as target


def reference() -> dict:
    return {
        "audit": {
            "status": "ok", "development_read": False,
            "blind_read": False, "training_scene_overlap": [],
            "scenes": 20, "episodes": 160,
        },
        "arms": {
            "certified_relocalization": {
                "joint": {"successes": 112},
                "revisit_given_novel_success": {
                    "eligible": 120, "successes": 112,
                },
            },
        },
        "certified_runtime": {
            "a_success_episodes": 120,
            "takeover_episodes": 115,
            "fallback_episodes_after_a_success": 5,
        },
    }


def dual() -> dict:
    return {
        "scope": {
            "development_or_blind_read": False,
            "train_scenes_only": True,
            "row_scene_held_out_for_cdec": True,
            "same_gpu_same_lingbot_process": True,
        },
        "method_gate": {
            "pass": True,
            "deployment_order": "geometry_first_then_cdec_on_reject",
            "requirements": {
                "at_least_one_certified_actionable_rescue": True,
                "cannot_lose_geometry_certified_actionable": True,
                "no_extra_certificate_false_positive": True,
                "same_anchor_certificate_decisions_repeatable": True,
            },
        },
        "policies": {
            "geometry": {
                "sessions": 480, "certificate_false_positive": 9,
            },
            "geometry_first_then_cdec_on_reject": {
                "sessions": 480, "certificate_false_positive": 9,
                "second_certificate_invocations": 349,
            },
        },
        "paired": {
            "geometry_first_cascade_minus_geometry_certified_actionable": {
                "gains": 1, "losses": 0,
            },
        },
        "proposal_identity": {},
    }


def independent(dual_payload: dict | None = None) -> dict:
    payload = dual() if dual_payload is None else dual_payload
    return {
        "verified": True,
        "scope": {
            "independent_of_primary_summarizer": True,
            "development_or_blind_read": False,
            "train40_only": True,
        },
        "inputs": {
            "official_report_sha256": target.EXPECTED_DUAL_SHA256,
        },
        "reconstructed": {
            field: copy.deepcopy(payload[field])
            for field in ("method_gate", "paired", "policies")
        } | {"proposal_identity": copy.deepcopy(
            payload.get("proposal_identity", {}))},
    }


def runtime_artifact() -> dict:
    return {
        "deployment_approved": False,
        "runtime_semantics": {
            "authority": "rank_frozen_causal_shortlist_only",
            "activation_authority": "independent_atomic_pnp_certificate",
            "accepted_geometry_can_be_overridden": False,
        },
    }


def protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "protocol.md"
    path.write_text("frozen protocol\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(target, "EXPECTED_PROTOCOL_SHA256", digest)
    return path


def test_exact_two_sided_mcnemar_needs_six_zero_loss_gains():
    assert target.exact_mcnemar(5, 0) == 0.0625
    assert target.exact_mcnemar(6, 0) == 0.03125
    assert target.minimum_zero_loss_gains(0.05) == 6


def test_reference_five_rejects_make_frozen_gate_unattainable(
        tmp_path, monkeypatch):
    report = target.build_audit(
        reference(), dual(), independent(), runtime_artifact(),
        reference_path=tmp_path / "reference.json",
        dual_path=tmp_path / "dual.json",
        independent_path=tmp_path / "independent.json",
        runtime_artifact_path=tmp_path / "runtime.json",
        protocol_path=protocol(tmp_path, monkeypatch),
    )
    gate = report["frozen_promotion_gate"]
    assert gate["best_case_available_gains"] == 5
    assert gate["minimum_zero_loss_gains_for_p_below_alpha"] == 6
    assert gate["best_case_exact_mcnemar_p"] == 0.0625
    assert gate["statistically_attainable"] is False
    assert report["resource_decision"][
        "submit_full_20_scene_160_episode_replay"] is False


def test_inconsistent_fallback_accounting_fails_closed(tmp_path, monkeypatch):
    payload = reference()
    payload["certified_runtime"]["fallback_episodes_after_a_success"] = 4
    with pytest.raises(RuntimeError, match="inconsistent"):
        target.build_audit(
            payload, dual(), independent(), runtime_artifact(),
            reference_path=tmp_path / "reference.json",
            dual_path=tmp_path / "dual.json",
            independent_path=tmp_path / "independent.json",
            runtime_artifact_path=tmp_path / "runtime.json",
            protocol_path=protocol(tmp_path, monkeypatch),
        )


def test_changed_dual_safety_result_fails_closed(tmp_path, monkeypatch):
    payload = copy.deepcopy(dual())
    payload["paired"][
        "geometry_first_cascade_minus_geometry_certified_actionable"
    ]["losses"] = 1
    with pytest.raises(RuntimeError, match="paired outcome changed"):
        target.build_audit(
            reference(), payload, independent(payload), runtime_artifact(),
            reference_path=tmp_path / "reference.json",
            dual_path=tmp_path / "dual.json",
            independent_path=tmp_path / "independent.json",
            runtime_artifact_path=tmp_path / "runtime.json",
            protocol_path=protocol(tmp_path, monkeypatch),
        )


def test_protocol_mutation_fails_closed(tmp_path, monkeypatch):
    path = protocol(tmp_path, monkeypatch)
    path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocol changed"):
        target.build_audit(
            reference(), dual(), independent(), runtime_artifact(),
            reference_path=tmp_path / "reference.json",
            dual_path=tmp_path / "dual.json",
            independent_path=tmp_path / "independent.json",
            runtime_artifact_path=tmp_path / "runtime.json",
            protocol_path=path,
        )


def test_independent_disagreement_fails_closed(tmp_path, monkeypatch):
    secondary = independent()
    secondary["reconstructed"]["paired"][
        "geometry_first_cascade_minus_geometry_certified_actionable"
    ]["gains"] = 2
    with pytest.raises(RuntimeError, match="secondary verifier disagrees"):
        target.build_audit(
            reference(), dual(), secondary, runtime_artifact(),
            reference_path=tmp_path / "reference.json",
            dual_path=tmp_path / "dual.json",
            independent_path=tmp_path / "independent.json",
            runtime_artifact_path=tmp_path / "runtime.json",
            protocol_path=protocol(tmp_path, monkeypatch),
        )


def test_runtime_artifact_cannot_self_approve(tmp_path, monkeypatch):
    artifact = runtime_artifact()
    artifact["deployment_approved"] = True
    with pytest.raises(RuntimeError, match="unexpectedly claims"):
        target.build_audit(
            reference(), dual(), independent(), artifact,
            reference_path=tmp_path / "reference.json",
            dual_path=tmp_path / "dual.json",
            independent_path=tmp_path / "independent.json",
            runtime_artifact_path=tmp_path / "runtime.json",
            protocol_path=protocol(tmp_path, monkeypatch),
        )
