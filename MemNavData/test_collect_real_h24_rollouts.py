import copy
import json
from pathlib import Path
import tempfile
import unittest

import MemNavData.test_real_h24_rollout_backend as backend_test_support

from MemNavData.collect_real_h24_rollouts import (
    CollectorError,
    GEOMETRY_MAP_SCHEMA,
    PLAN_DIAGNOSTICS_SCHEMA,
    _atomic_write_pair,
    assert_pythonpath,
    build_plan_diagnostics,
    build_run_signature,
    canonical_bytes,
    decision_seeds,
    load_candidate_records,
    load_geometry_map,
    load_plan_diagnostics,
    parse_args,
    safe_state_stem,
    selected_shard_records,
    validate_resume_pair,
)
from MemNavData.real_h24_rollout_backend import (
    PurePursuitConfig,
    RealH24RolloutBackend,
)
from MemNavData.habitat_rollout_primitives import (
    FrozenGeometryIdentity,
    NAVMESH_BOOL_FIELDS,
    NAVMESH_FLOAT_FIELDS,
)
from MemNavData.novel_rollout_protocol_v2 import (
    CandidateArm,
    atomic_write_artifact,
    collect_paired_rollouts,
)
from MemNavData.test_novel_candidate_set_schema_v2 import record
from MemNavData.test_novel_rollout_protocol_v2 import (
    NATIVE,
    RESIDUAL,
    SEEDS,
    factory,
    state,
)


def sha256_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def neutral_precollection(value):
    value = copy.deepcopy(value)
    for candidate in value["candidates"]:
        labels = candidate["labels"]
        for key in (
            "geodesic_progress_h8_m",
            "geodesic_progress_h24_m",
            "advantage_h24_m",
        ):
            labels[key] = 0.0
        for key in (
            "harm", "useful", "reachable", "collision_h8",
            "regression_h24", "rollout_label_valid",
        ):
            labels[key] = False
    value["set_labels"].update({
        "candidate_set_has_positive": False,
        "candidate_universe_has_positive": False,
        "candidate_coverage_miss": False,
        "coverage_label_valid": False,
        "oracle_best_candidate_id": "dustbin",
    })
    return value


def plan_diagnostics(artifact):
    trajectory = [
        [0.1 * (index + 1), 0.0, 0.01 * (index + 1)]
        for index in range(24)
    ]
    rows = {}
    for outcome in artifact.outcomes:
        plans = {}
        for plan in outcome.plans:
            plans[plan.plan_sha256] = {
                "plan_sha256": plan.plan_sha256,
                "server_selected_trajectory_index": 0,
                "raw_selected_trajectory": trajectory,
                "executable_trajectory": trajectory,
                "all_trajectory": [[trajectory]],
                "all_values": [[1.0]],
                "critic_max": 1.0,
                "stop_threshold": -0.5,
                "low_critic_fallback_applied": False,
                "behaviorally_identical_xy": True,
                "server_receipt_sha256": "e" * 64,
            }
        rows[outcome.candidate_id] = plans
    return {
        "schema_version": PLAN_DIAGNOSTICS_SCHEMA,
        "artifact_sha256": artifact.artifact_sha256,
        "run_signature_sha256": artifact.run_signature_sha256,
        "state_id": artifact.state.state_id,
        "diffusion_seeds": list(artifact.diffusion_seeds),
        "by_candidate": rows,
    }


class RealH24CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path, sha256_bytes(raw)

    def test_canonical_json_and_jsonl_accept_neutral_precollection(self):
        first = neutral_precollection(record(
            state_id="scene-a/episode-0/state-0", positive=True))
        second = neutral_precollection(record(
            plan_index=1,
            state_id="scene-a/episode-0/state-1",
            positive=True,
        ))
        json_path, json_sha = self.write(
            "candidates.json", canonical_bytes([first, second]))
        loaded, neutral = load_candidate_records(json_path, json_sha)
        self.assertEqual(loaded, [first, second])
        self.assertTrue(neutral)

        jsonl = canonical_bytes(first) + canonical_bytes(second)
        jsonl_path, jsonl_sha = self.write("candidates.jsonl", jsonl)
        loaded_jsonl, neutral_jsonl = load_candidate_records(
            jsonl_path, jsonl_sha)
        self.assertEqual(loaded_jsonl, [first, second])
        self.assertTrue(neutral_jsonl)

        labeled = record(state_id="scene-a/episode-0/labeled")
        labeled_path, labeled_sha = self.write(
            "labeled.json", canonical_bytes(labeled))
        labeled_rows, labeled_neutral = load_candidate_records(
            labeled_path, labeled_sha)
        self.assertEqual(labeled_rows, [labeled])
        self.assertFalse(labeled_neutral)

    def test_candidate_input_fails_on_noncanonical_or_fake_neutral_labels(self):
        value = neutral_precollection(record())
        noncanonical = (json.dumps(value, indent=2) + "\n").encode()
        path, digest = self.write("bad.json", noncanonical)
        with self.assertRaisesRegex(CollectorError, "not canonical"):
            load_candidate_records(path, digest)

        value["candidates"][1]["labels"]["advantage_h24_m"] = 0.1
        raw = canonical_bytes(value)
        path, digest = self.write("fake-neutral.json", raw)
        with self.assertRaisesRegex(CollectorError, "invalid"):
            load_candidate_records(path, digest)

    def test_sharding_and_seed_derivation_are_order_independent(self):
        rows = [record(
            plan_index=index,
            state_id=f"scene-a/episode-0/state-{index}",
        ) for index in range(7)]
        first = selected_shard_records(rows, 1, 3)
        reverse = selected_shard_records(list(reversed(rows)), 1, 3)
        self.assertEqual(
            [row["provenance"]["state_id"] for row in first],
            [row["provenance"]["state_id"] for row in reverse],
        )
        self.assertEqual(len(first), 2)
        self.assertEqual(
            decision_seeds(17, "state-a"), decision_seeds(17, "state-a"))
        self.assertEqual(len(set(decision_seeds(17, "state-a"))), 3)
        self.assertNotEqual(
            decision_seeds(17, "state-a"), decision_seeds(17, "state-b"))
        self.assertNotEqual(safe_state_stem("a/b"), safe_state_stem("a_b"))

    def test_geometry_map_is_canonical_pinned_and_cannot_escape(self):
        glb = self.root / "scene.glb"
        navmesh = self.root / "scene.navmesh"
        glb.write_bytes(b"glb")
        navmesh.write_bytes(b"navmesh")
        settings = {name: 1.0 for name in NAVMESH_FLOAT_FIELDS}
        settings.update({name: False for name in NAVMESH_BOOL_FIELDS})
        settings["agent_radius"] = 0.3
        settings["agent_height"] = 1.5
        identity = FrozenGeometryIdentity.capture(
            glb_path=glb,
            navmesh_path=navmesh,
            habitat_sim_version="0.3.3",
            agent_radius_m=0.3,
            agent_height_m=1.5,
            navmesh_settings=settings,
        )
        identity_path = self.root / "identities/scene.json"
        identity.write_json(identity_path)
        mapping = {
            "schema_version": GEOMETRY_MAP_SCHEMA,
            "scenes": {
                "scene": {
                    "identity_path": "identities/scene.json",
                    "identity_sha256": identity.identity_sha256,
                },
            },
        }
        path, digest = self.write("geometry-map.json", canonical_bytes(mapping))
        loaded = load_geometry_map(path, digest)
        self.assertEqual(loaded["scene"].identity, identity)

        mapping["scenes"]["scene"]["identity_path"] = "../escape.json"
        raw = canonical_bytes(mapping)
        bad_path, bad_digest = self.write("bad-map.json", raw)
        with self.assertRaisesRegex(CollectorError, "escapes"):
            load_geometry_map(bad_path, bad_digest)

    def test_resume_requires_complete_strict_artifact_and_diagnostics(self):
        fixture = backend_test_support.RealH24RolloutBackendTests(
            methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        transport = backend_test_support.FakeTransport()
        runtime = backend_test_support.FakeRuntime(fixture.identity)
        backends = {}

        def real_factory(candidate_id):
            backend = RealH24RolloutBackend(
                fixture.assets,
                transport,
                runtime,
                expected_server_provenance=(
                    backend_test_support.FakeTransport.provenance),
                stop_threshold=-0.5,
            )
            backends[candidate_id] = backend
            return backend

        artifact = collect_paired_rollouts(
            real_factory,
            fixture.assets.state,
            (
                CandidateArm("native", "native"),
                CandidateArm("frontier-1", "frontier", (1.0, -3.0)),
            ),
            SEEDS,
            run_signature_sha256="d" * 64,
        )
        artifact_path = self.root / "state.json"
        diagnostics_path = self.root / "state.plans.json"
        atomic_write_artifact(artifact_path, artifact)
        _atomic_write_pair(
            diagnostics_path,
            build_plan_diagnostics(artifact, backends),
        )
        resumed = validate_resume_pair(
            artifact_path,
            diagnostics_path,
            state_id=artifact.state.state_id,
            run_signature_sha256=artifact.run_signature_sha256,
            diffusion_seeds=artifact.diffusion_seeds,
            candidate_ids=["native", "frontier-1"],
        )
        self.assertEqual(resumed.artifact_sha256, artifact.artifact_sha256)
        loaded = load_plan_diagnostics(diagnostics_path)
        self.assertEqual(loaded["artifact_sha256"], artifact.artifact_sha256)
        self.assertTrue(all(
            row["behaviorally_identical_xy"]
            for plans in loaded["by_candidate"].values()
            for row in plans.values()
        ))

        bad_equivalence = build_plan_diagnostics(artifact, backends)
        residual_plans = bad_equivalence["by_candidate"]["frontier-1"]
        next(iter(residual_plans.values()))[
            "behaviorally_identical_xy"] = False
        bad_equivalence_path = self.root / "bad-equivalence.plans.json"
        _atomic_write_pair(bad_equivalence_path, bad_equivalence)
        with self.assertRaisesRegex(CollectorError, "behavioral XY"):
            validate_resume_pair(
                artifact_path,
                bad_equivalence_path,
                state_id=artifact.state.state_id,
                run_signature_sha256=artifact.run_signature_sha256,
                diffusion_seeds=artifact.diffusion_seeds,
                candidate_ids=["native", "frontier-1"],
            )

        tampered = build_plan_diagnostics(artifact, backends)
        candidate_plans = next(iter(tampered["by_candidate"].values()))
        diagnostic = next(iter(candidate_plans.values()))
        diagnostic["all_values"] = [[2.0]]
        diagnostic["critic_max"] = 2.0
        tampered_path = self.root / "tampered.plans.json"
        _atomic_write_pair(tampered_path, tampered)
        with self.assertRaisesRegex(CollectorError, "reproduce artifact plan hash"):
            validate_resume_pair(
                artifact_path,
                tampered_path,
                state_id=artifact.state.state_id,
                run_signature_sha256=artifact.run_signature_sha256,
                diffusion_seeds=artifact.diffusion_seeds,
                candidate_ids=["native", "frontier-1"],
            )

        diagnostics_path.with_suffix(
            diagnostics_path.suffix + ".sha256").unlink()
        with self.assertRaisesRegex(CollectorError, "complete"):
            validate_resume_pair(
                artifact_path,
                diagnostics_path,
                state_id=artifact.state.state_id,
                run_signature_sha256=artifact.run_signature_sha256,
                diffusion_seeds=artifact.diffusion_seeds,
                candidate_ids=["native", "frontier-1"],
            )

    def test_diagnostics_mark_xy_difference_relative_to_native(self):
        fixture = backend_test_support.RealH24RolloutBackendTests(
            methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        transport = backend_test_support.FakeTransport(
            image_point_x_offset=0.2)
        runtime = backend_test_support.FakeRuntime(fixture.identity)
        backends = {}

        def real_factory(candidate_id):
            backend = RealH24RolloutBackend(
                fixture.assets,
                transport,
                runtime,
                expected_server_provenance=(
                    backend_test_support.FakeTransport.provenance),
                stop_threshold=-0.5,
            )
            backends[candidate_id] = backend
            return backend

        artifact = collect_paired_rollouts(
            real_factory,
            fixture.assets.state,
            (
                CandidateArm("native", "native"),
                CandidateArm("frontier-1", "frontier", (1.0, -3.0)),
            ),
            SEEDS,
            run_signature_sha256="d" * 64,
        )
        diagnostics = build_plan_diagnostics(artifact, backends)
        self.assertTrue(all(
            row["behaviorally_identical_xy"]
            for row in diagnostics["by_candidate"]["native"].values()
        ))
        self.assertTrue(all(
            not row["behaviorally_identical_xy"]
            for row in diagnostics["by_candidate"]["frontier-1"].values()
        ))

    def test_diagnostics_reject_unknown_fields_and_nonfinite_values(self):
        artifact = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256="d" * 64,
        )
        payload = plan_diagnostics(artifact)
        first_candidate = next(iter(payload["by_candidate"].values()))
        first_plan = next(iter(first_candidate.values()))
        first_plan["unknown"] = 1
        path = self.root / "unknown.plans.json"
        _atomic_write_pair(path, payload)
        with self.assertRaisesRegex(CollectorError, "schema/fields"):
            load_plan_diagnostics(path)

        payload = plan_diagnostics(artifact)
        first_candidate = next(iter(payload["by_candidate"].values()))
        first_plan = next(iter(first_candidate.values()))
        first_plan["all_values"] = [float("nan")]
        with self.assertRaisesRegex(CollectorError, "JSON-compatible"):
            _atomic_write_pair(self.root / "nan.plans.json", payload)

    def test_pythonpath_self_check(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(assert_pythonpath(root), root)
        with self.assertRaisesRegex(CollectorError, "absent from PYTHONPATH"):
            assert_pythonpath(self.root / "not-on-path")

    def test_run_signature_binds_explicit_legacy_camera_height(self):
        common = {
            "candidate_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "geometry_map_sha256": "3" * 64,
            "server_provenance_sha256": "4" * 64,
            "server_url": "http://127.0.0.1:18888",
            "base_seed": 17,
            "stop_threshold": -0.5,
            "controller": PurePursuitConfig(),
        }
        first = build_run_signature(
            **common, legacy_camera_height_m=0.5)
        second = build_run_signature(
            **common, legacy_camera_height_m=0.6)
        self.assertNotEqual(first, second)

    def test_cli_requires_explicit_stop_threshold(self):
        argv = [
            "--candidate-sets", "candidates.json",
            "--expected-candidate-sha256", "1" * 64,
            "--expert-manifest", "manifest.json",
            "--expected-manifest-sha256", "2" * 64,
            "--geometry-map", "geometry.json",
            "--expected-geometry-map-sha256", "3" * 64,
            "--server-provenance", "server.json",
            "--expected-server-provenance-sha256", "4" * 64,
            "--server-url", "http://127.0.0.1:18888",
            "--output-root", "output",
            "--legacy-camera-height-m", "0.5",
        ]
        with self.assertRaises(SystemExit):
            parse_args(argv)


if __name__ == "__main__":
    unittest.main()
