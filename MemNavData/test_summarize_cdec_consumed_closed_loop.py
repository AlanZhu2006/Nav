import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.summarize_cdec_consumed_closed_loop import (
    audit_episode,
    cdec_neutral_payload,
    learned_takeover,
    promotion_decision,
)


class CDECConsumedClosedLoopSummaryTest(unittest.TestCase):
    @staticmethod
    def _metric(*, reached_b: bool) -> dict[str, str]:
        return {
            "seed": "123",
            "leg1_trace_sha256": "a" * 64,
            "reached_A": "1.0",
            "spl_A": "0.8",
            "geo_A": "3.0",
            "len_A": "3.5",
            "final_dist_A": "0.4",
            "steps_A": "12",
            "termination_reason_A": "success",
            "blocked_steps_A": "0",
            "reached_B": "1.0" if reached_b else "0.0",
            "spl_B": "0.7" if reached_b else "0.0",
            "spl_B_with_terminal": "0.7" if reached_b else "0.0",
            "geo_B": "4.0",
            "steps_B": "20" if reached_b else "50",
            "steps_B_diagnostic": "20" if reached_b else "50",
            "steps_B_at_reach": "20" if reached_b else "",
            "len_B": "4.5" if reached_b else "9.0",
            "len_B_at_reach": "4.5" if reached_b else "",
            "final_dist_B": "0.5" if reached_b else "2.0",
            "termination_reason_B": "success" if reached_b else "budget",
            "blocked_steps_B": "0",
            "blocked_step_rate_B": "0.0",
            "terminal_final_goal_dist_m": "0.5" if reached_b else "2.0",
        }

    @staticmethod
    def _plan(*, requested: bool, learned_accepted: bool = False) -> dict:
        geometry = {
            "source": "geometry",
            "selected_anchor": 7,
            "accepted": False,
            "reason": "rejected",
        }
        attempts = [geometry]
        learned = {
            "status": "not_requested",
            "activation_authorized": False,
        }
        source = "geometry"
        accepted = False
        takeover = False
        if requested:
            learned = {
                "status": (
                    "certificate_accepted" if learned_accepted
                    else "certificate_rejected"
                ),
                "activation_authorized": False,
            }
            attempts.append({
                "source": "learned_on_geometry_reject",
                "selected_anchor": 9,
                "accepted": learned_accepted,
                "reason": "certificate_accepted" if learned_accepted else "rejected",
            })
            if learned_accepted:
                source = "learned_on_geometry_reject"
                accepted = True
                takeover = True
        return {
            "step": 0,
            "requested_diffusion_seed": 123,
            "diffusion_seed": 123,
            "frame_idx": 20,
            "goal_start_frame": 20,
            "candidate_ceiling": 20,
            "router_candidate_order_dino": [7, 9],
            "router_candidate_order_used": [7, 9],
            "router_ranking_mode": (
                "cascade" if requested else "geometry"
            ),
            "certified_relocalization_learned_rescue_requested": requested,
            "certified_relocalization_proposal_attempts": attempts,
            "certified_relocalization_learned_proposal": learned,
            "certified_relocalization_selected_proposal_source": source,
            "certified_relocalization_accepted": accepted,
            "revisit_adapter_takeover": takeover,
        }

    def test_episode_audit_distinguishes_rejected_probe_from_takeover(self):
        trace_sha = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry_root = root / "geometry"
            cdec_root = root / "cdec"
            geometry_root.mkdir()
            cdec_root.mkdir()
            common = {
                "leg1_trace_sha256": trace_sha,
                "legA": [{"step": 0, "trajectory": [[0.0, 0.0]]}],
                "legA_memory_trace": [{"frame_idx": 0, "sha256": "b" * 64}],
            }
            (geometry_root / "episode_0000_plans.json").write_text(json.dumps({
                **common, "legB": [self._plan(requested=False)],
            }))
            (cdec_root / "episode_0000_plans.json").write_text(json.dumps({
                **common,
                "legB": [self._plan(requested=True, learned_accepted=False)],
            }))
            metric = self._metric(reached_b=False)
            audit = audit_episode(
                scene="scene", episode="episode_0000",
                geometry_root=geometry_root, cdec_root=cdec_root,
                geometry_metric=metric, cdec_metric=dict(metric),
                trace_sha=trace_sha,
            )
            self.assertTrue(audit["learned_invoked"])
            self.assertFalse(audit["learned_takeover"])
            self.assertTrue(audit["no_treatment_exact"])

            (cdec_root / "episode_0000_plans.json").write_text(json.dumps({
                **common,
                "legB": [self._plan(requested=True, learned_accepted=True)],
            }))
            audit = audit_episode(
                scene="scene", episode="episode_0000",
                geometry_root=geometry_root, cdec_root=cdec_root,
                geometry_metric=metric,
                cdec_metric=self._metric(reached_b=True),
                trace_sha=trace_sha,
            )
            self.assertTrue(audit["learned_takeover"])
            self.assertFalse(audit["no_treatment_exact"])

    def test_neutral_payload_removes_only_timing_and_cdec_arm_labels(self):
        first = {
            "step": 1,
            "certified_relocalization_ms": 10.0,
            "router_ranking_mode": "geometry",
            "certified_relocalization_learned_rescue_requested": False,
            "certified_relocalization_learned_proposal": {"status": "off"},
            "certified_relocalization_proposal_attempts": [
                {"source": "geometry", "accepted": False, "anchor": 7}
            ],
            "certificate": {"accepted": True},
        }
        second = {
            **first,
            "certified_relocalization_ms": 99.0,
            "router_ranking_mode": "cascade",
            "certified_relocalization_learned_rescue_requested": True,
            "certified_relocalization_learned_proposal": {
                "status": "not_evaluated_geometry_accepted"
            },
            "certified_relocalization_proposal_attempts": [
                {"source": "geometry", "accepted": False, "anchor": 7},
                {
                    "source": "learned_on_geometry_reject",
                    "accepted": False,
                    "anchor": 9,
                },
            ],
        }
        self.assertEqual(cdec_neutral_payload(first), cdec_neutral_payload(second))
        second["certified_relocalization_proposal_attempts"][0]["anchor"] = 8
        self.assertNotEqual(cdec_neutral_payload(first), cdec_neutral_payload(second))
        second["certified_relocalization_proposal_attempts"][0]["anchor"] = 7
        second["certificate"] = {"accepted": False}
        self.assertNotEqual(cdec_neutral_payload(first), cdec_neutral_payload(second))

    def test_takeover_requires_source_certificate_and_adapter(self):
        plan = {
            "certified_relocalization_selected_proposal_source": (
                "learned_on_geometry_reject"
            ),
            "certified_relocalization_accepted": True,
            "revisit_adapter_takeover": True,
        }
        self.assertTrue(learned_takeover(plan))
        plan["certified_relocalization_accepted"] = False
        self.assertFalse(learned_takeover(plan))

    def test_promotion_gate_is_strict_and_frozen(self):
        gains = [
            {"scene": "a", "learned_takeover": True},
            {"scene": "b", "learned_takeover": True},
        ]
        passed = promotion_decision(
            gains=gains,
            losses=[],
            mcnemar_p=0.03125,
            cluster_interval_95=[0.01, 0.2],
            all_audits_pass=True,
        )
        self.assertTrue(passed["pass"])
        self.assertFalse(
            passed["authorize_blind_opening_without_explicit_user_approval"]
        )
        for mutation in (
            dict(gains=[gains[0]]),
            dict(losses=[{"scene": "c", "learned_takeover": True}]),
            dict(mcnemar_p=0.05),
            dict(cluster_interval_95=[0.0, 0.2]),
            dict(all_audits_pass=False),
            dict(gains=[
                {"scene": "a", "learned_takeover": True},
                {"scene": "b", "learned_takeover": False},
            ]),
        ):
            args = {
                "gains": gains,
                "losses": [],
                "mcnemar_p": 0.03125,
                "cluster_interval_95": [0.01, 0.2],
                "all_audits_pass": True,
            }
            args.update(mutation)
            self.assertFalse(promotion_decision(**args)["pass"])


if __name__ == "__main__":
    unittest.main()
