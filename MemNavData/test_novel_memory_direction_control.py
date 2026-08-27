import json
import math
from pathlib import Path
import tempfile
import unittest

from novel_memory_direction_control import (
    ARMS,
    RandomizedBearingAdapter,
    SCHEMA_VERSION,
    deterministic_random_bearing,
    replay_deranged_sidecar,
    sha256_file,
    validate_control_manifest,
)
from revisit_bearing_adapter import adapt_revisit_pointgoal
from freeze_novel_memory_direction_control import assign_donors
from summarize_novel_memory_direction_control import summarize
from independent_verify_novel_memory_direction_control import verify


class NovelMemoryDirectionControlTest(unittest.TestCase):
    def test_random_bearing_is_deterministic_bounded_and_identity_sensitive(self):
        kwargs = dict(
            global_seed=20260816,
            scene="scene_a",
            episode="episode_0001",
            plan_index=3,
        )
        first = deterministic_random_bearing(**kwargs)
        second = deterministic_random_bearing(**kwargs)
        changed = deterministic_random_bearing(**{**kwargs, "plan_index": 4})
        self.assertEqual(first, second)
        self.assertEqual(
            first["key_sha256"],
            "3c481602c25c6411e2b36cc2ac9acf494c479c66a186e9e8928d9f255ac250e6",
        )
        self.assertAlmostEqual(first["angle_rad"], -1.6620599404829537, places=15)
        self.assertNotEqual(first["key_sha256"], changed["key_sha256"])
        self.assertGreaterEqual(first["angle_rad"], -math.pi)
        self.assertLess(first["angle_rad"], math.pi)
        self.assertAlmostEqual(
            math.hypot(*first["unit_bearing"]), 1.0, places=12
        )

    def test_randomized_adapter_preserves_availability_and_fixed_radius(self):
        wrapper = RandomizedBearingAdapter(
            original_adapter=adapt_revisit_pointgoal,
            global_seed=7,
            scene="s",
            episode="e",
            query_id="q",
        )
        decision = wrapper(
            mode="raw_fixed_bearing_v1",
            router_active=True,
            pointgoal=[3.0, 4.0],
            source="raw",
            pointgoal_units="metric_m",
        )
        self.assertTrue(decision.takeover)
        self.assertAlmostEqual(
            math.hypot(*decision.controller_pointgoal), 2.5, places=12
        )
        self.assertEqual(len(wrapper.ledger), 1)
        audit = wrapper.ledger[0]
        self.assertEqual(audit["factual_unit_bearing"], [0.6, 0.8])
        self.assertTrue(audit["factual_takeover"])
        self.assertTrue(audit["randomized_takeover"])
        self.assertNotEqual(
            audit["factual_unit_bearing"], audit["randomized_unit_bearing"]
        )

    def test_randomized_adapter_does_not_create_an_unavailable_proposal(self):
        wrapper = RandomizedBearingAdapter(
            original_adapter=adapt_revisit_pointgoal,
            global_seed=7,
            scene="s",
            episode="e",
            query_id="q",
        )
        decision = wrapper(
            mode="raw_fixed_bearing_v1",
            router_active=True,
            pointgoal=None,
            source="raw",
            pointgoal_units="metric_m",
        )
        self.assertFalse(decision.takeover)
        self.assertFalse(wrapper.ledger[0]["raw_proposal_available"])
        self.assertIsNone(wrapper.ledger[0]["randomized_angle_rad"])

    def test_randomized_adapter_rejects_the_wrong_runtime_mode(self):
        wrapper = RandomizedBearingAdapter(
            original_adapter=adapt_revisit_pointgoal,
            global_seed=7,
            scene="s",
            episode="e",
            query_id="q",
        )
        with self.assertRaises(RuntimeError):
            wrapper(
                mode="verified_bearing_v1",
                router_active=True,
                pointgoal=[1.0, 0.0],
            )

    @staticmethod
    def _source(root: Path, name: str, frame_count: int, plan_steps: list[int]):
        source = root / name
        (source / "rgb").mkdir(parents=True)
        poses = []
        for step in range(frame_count):
            image = source / "rgb" / f"{step:06d}.jpg"
            image.write_bytes(f"{name}-{step}".encode())
            poses.append(
                {
                    "step": step,
                    "jpg_sha256": sha256_file(image),
                    "x": float(step),
                    "z": 0.0,
                    "yaw": 0.0,
                }
            )
        trace = {
            "reached": True,
            "poses": poses,
            "plans": [{"step": step} for step in plan_steps],
        }
        (source / "online_a_trace.json").write_text(json.dumps(trace))
        (source / "receipt.json").write_text(json.dumps({"name": name}))
        return {
            "source": source,
            "trace": trace,
            "receipt": {"name": name},
            "scene": name,
            "episode": "episode_0000",
        }

    def test_deranged_replay_separates_sidecar_from_factual_fifo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            factual = self._source(root, "factual", 4, [0, 2])
            donor = self._source(root, "donor", 3, [0, 1])
            memory_payloads = []
            fifo_payloads = []

            def memory_step(payload):
                memory_payloads.append(payload)
                return {"frame_idx": len(memory_payloads) - 1}

            def navdp_step(payload):
                fifo_payloads.append(payload)
                return {
                    "diffusion_sampled": False,
                    "memory_size": 8,
                    "queue_lengths": [len(fifo_payloads)],
                }

            receipt = replay_deranged_sidecar(
                factual,
                donor,
                memory_step=memory_step,
                navdp_replay_step=navdp_step,
            )
            self.assertEqual(memory_payloads, [b"donor-0", b"donor-1", b"donor-2"])
            self.assertEqual(fifo_payloads, [b"factual-0", b"factual-2"])
            self.assertEqual(receipt["online_frames"], 4)
            self.assertEqual(receipt["sidecar_memory_frames"], 3)
            self.assertTrue(receipt["sidecar_is_deranged"])

    def test_manifest_rejects_fixed_points_and_accepts_a_permutation(self):
        rows = [
            {
                "scene": "a",
                "episode": "e0",
                "donor": {"scene": "b", "episode": "e1"},
                "arm_order": list(ARMS),
            },
            {
                "scene": "b",
                "episode": "e1",
                "donor": {"scene": "a", "episode": "e0"},
                "arm_order": list(ARMS[1:] + ARMS[:1]),
            },
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "confirmation_claim_allowed": False,
            "query_role": "novel",
            "episodes": rows,
        }
        validate_control_manifest(payload)
        rows[0]["donor"] = {"scene": "a", "episode": "e0"}
        with self.assertRaises(RuntimeError):
            validate_control_manifest(payload)

    def test_donor_assignment_prefers_same_scene_and_is_a_permutation(self):
        rows = [
            {"scene": "a", "episode": "e0", "decision_frames": 5},
            {"scene": "a", "episode": "e1", "decision_frames": 7},
            {"scene": "b", "episode": "e0", "decision_frames": 10},
            {"scene": "c", "episode": "e0", "decision_frames": 11},
        ]
        donors = assign_donors(rows)
        self.assertEqual(donors[("a", "e0")]["scene"], "a")
        self.assertEqual(donors[("a", "e1")]["scene"], "a")
        identities = {(row["scene"], row["episode"]) for row in rows}
        donor_identities = {
            (row["scene"], row["episode"]) for row in donors.values()
        }
        self.assertEqual(identities, donor_identities)
        self.assertTrue(all(identity != (
            donors[identity]["scene"], donors[identity]["episode"]
        ) for identity in identities))

    def test_singleton_is_merged_without_a_fixed_point(self):
        rows = [
            {"scene": "a", "episode": "e0", "decision_frames": 5},
            {"scene": "a", "episode": "e1", "decision_frames": 7},
            {"scene": "b", "episode": "e0", "decision_frames": 6},
        ]
        donors = assign_donors(rows)
        self.assertEqual(len(donors), 3)
        for identity, donor in donors.items():
            self.assertNotEqual(identity, (donor["scene"], donor["episode"]))

    def test_summary_and_independent_verifier_recompute_raw_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "control_manifest.json"
            arm_orders = [
                list(ARMS), list(ARMS[1:] + ARMS[:1]),
            ]
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "evaluation_stage": "consumed_development_mechanism_only",
                "confirmation_claim_allowed": False,
                "query_role": "novel",
                "arms": list(ARMS),
                "untouched_final_scenes_remain_unread": ["final_scene"],
                "episodes": [
                    {
                        "scene": "a", "episode": "episode_0000",
                        "arm_order": arm_orders[0],
                        "donor": {"scene": "b", "episode": "episode_0001"},
                    },
                    {
                        "scene": "b", "episode": "episode_0001",
                        "arm_order": arm_orders[1],
                        "donor": {"scene": "a", "episode": "episode_0000"},
                    },
                ],
            }
            manifest_path.write_text(json.dumps(manifest))
            manifest_sha = sha256_file(manifest_path)
            outcomes = [
                {
                    "native": 0, "raw_factual_history": 1,
                    "raw_deranged_history": 0, "raw_randomized_bearing": 0,
                },
                {
                    "native": 1, "raw_factual_history": 1,
                    "raw_deranged_history": 1, "raw_randomized_bearing": 1,
                },
            ]
            for index, row in enumerate(manifest["episodes"]):
                directory = (
                    root / "evaluation"
                    / f"{index:03d}_{row['scene']}_{row['episode']}"
                )
                directory.mkdir(parents=True)
                (directory / "episode_contract.json").write_text(json.dumps({
                    "selection_index": index,
                    "arm_order": row["arm_order"],
                    "confirmation_claim_allowed": False,
                }))
                completion = {
                    "schema_version": (
                        "novel_memory_direction_completion_v1_20260816"
                    ),
                    "control_manifest_sha256": manifest_sha,
                    "confirmation_claim_allowed": False,
                    "prefix_equality": True,
                    "factual_fifo_equality": True,
                    "deranged_sidecar_verified": True,
                    "randomized_bearing_verified": True,
                    "zero_takeover_exact_fallback_verified": True,
                    "scene": row["scene"], "episode": row["episode"],
                    "query_id": "pair_00_novel",
                    "outcomes": outcomes[index],
                    "takeover_plans": {arm: int(arm != "native") for arm in ARMS},
                    "fallback_plans": {arm: int(arm == "native") for arm in ARMS},
                    "plan_count": {arm: 1 for arm in ARMS},
                    "geodesic_m": {arm: 2.0 for arm in ARMS},
                    "path_length_m": {arm: 4.0 for arm in ARMS},
                    "final_distance_m": {arm: 0.8 for arm in ARMS},
                    "final_geodesic_m": {arm: 0.9 for arm in ARMS},
                    "steps": {arm: 8 for arm in ARMS},
                    "spl": {
                        arm: outcomes[index][arm] * 0.5 for arm in ARMS
                    },
                    "wall_time_seconds": {arm: 1.0 for arm in ARMS},
                }
                completion_path = directory / "completion.json"
                completion_path.write_text(json.dumps(completion))
                (directory / "completion.json.sha256").write_text(
                    f"{sha256_file(completion_path)}  completion.json\n"
                )
                for arm in ARMS:
                    arm_dir = directory / arm
                    arm_dir.mkdir()
                    with (arm_dir / "metric.csv").open("w", newline="") as handle:
                        fields = [
                            "scene", "episode", "query_id", "analysis_role",
                            "arm", "reached", "geodesic_m", "path_len_m",
                            "final_goal_dist_m", "final_goal_geodesic_m",
                            "steps", "adapter_takeover_plans",
                        ]
                        writer = __import__("csv").DictWriter(
                            handle, fieldnames=fields
                        )
                        writer.writeheader()
                        writer.writerow({
                            "scene": row["scene"], "episode": row["episode"],
                            "query_id": "pair_00_novel", "analysis_role": "novel",
                            "arm": arm, "reached": outcomes[index][arm],
                            "geodesic_m": 2.0, "path_len_m": 4.0, "steps": 8,
                            "final_goal_dist_m": 0.8,
                            "final_goal_geodesic_m": 0.9,
                            "adapter_takeover_plans": int(arm != "native"),
                        })
                    is_deranged = arm == "raw_deranged_history"
                    sidecar_hash = (
                        f"deranged-{index}" if is_deranged else f"factual-{index}"
                    )
                    ledger = ([{
                        "factual_takeover": True,
                        "randomized_takeover": True,
                    }] if arm == "raw_randomized_bearing" else [])
                    plan = {
                        "arm": arm,
                        "analysis_role_not_forwarded": True,
                        "query_leg": [{
                            "requested_diffusion_seed": 17,
                            "diffusion_seed": 17,
                        }],
                        "rollout_traces": {"query": [{"step": 0}]},
                        "replay": {
                            "factual_fifo_decision_sha256": f"fifo-{index}",
                            "sidecar_is_deranged": is_deranged,
                            "sidecar_memory_sha256": sidecar_hash,
                        },
                        "novel_causal_control": {
                            "arm": arm,
                            "manifest_sha256": manifest_sha,
                            "randomized_bearing_ledger": ledger,
                        },
                    }
                    (arm_dir / f"{row['episode']}_pair_00_novel_plans.json").write_text(
                        json.dumps(plan)
                    )
            summary_path = root / "summary.json"
            result = summarize(root, manifest_path, summary_path)
            self.assertEqual(
                result["arm_metrics"]["raw_factual_history"]["successes"], 2
            )
            checked = verify(root, manifest_path, summary_path)
            self.assertTrue(checked["verified"])
            self.assertEqual(checked["population"], {"episodes": 2, "scenes": 2})


if __name__ == "__main__":
    unittest.main()
