"""Pure-fixture tests for the manifest-native causal geometry teacher.

The fixture deliberately has one scene, two physical three-leg episodes, and
the six source states needed to cover Goal-B t0/midpoint and Goal-C,
factual/counterfactual.  DINO is only a deterministic shortlist provider and
the renderer is a deterministic depth sensor; neither supplies teacher labels.
"""

from __future__ import annotations

import builtins
import copy
import csv
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from MemNavData.build_manifest_causal_covisibility_teacher import (
    ARTIFACT_NAME,
    AUDIT_NAME,
    BUNDLE_FILES,
    CausalTeacherError,
    CSV_NAME,
    DINO_IDENTITY_SCHEMA,
    EMBEDDING_RECEIPT_NAME,
    MANIFEST_SCHEMA,
    M_W,
    PinnedDINOEmbeddingBundleProvider,
    RENDERER_IDENTITY_SCHEMA,
    TeacherConfig,
    _exclusive_stage_writer,
    _goal_camera_pose,
    _phase_b_kind,
    _physical_runtime_record,
    _scene_grouped_work_items,
    _verify_static_runtime_records,
    build_dino_embedding_bundle,
    build_teacher_artifact,
    canonical_json_bytes,
    make_provider_identity,
    sha256_bytes,
    sha256_file,
    teacher_bundle_payloads,
    temporal_nms_shortlist,
    write_teacher_bundle,
)
from MemNavData.build_novel_candidate_manifest import PARQUET_PREFIX_COLUMNS


SCENE = "fixture_scene"
SOURCE_EPISODE = "episode_0000"
PARTNER_EPISODE = "episode_0001"
N_FRAMES = 6
IMAGE_SIZE = 8


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "path_sha256": sha256_bytes(relative.encode("utf-8")),
        "bytes": path.stat().st_size,
        "content_sha256": sha256_file(path),
    }


def _sequence_sha(values: Sequence[object]) -> str:
    return sha256_bytes(canonical_json_bytes(list(values)))


class FakeDINO:
    """Exact-shaped deterministic embeddings with known top frames."""

    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.identity = make_provider_identity(
            DINO_IDENTITY_SCHEMA,
            "fixture_exact_lingbot_dino_cls",
            {"checkpoint_content_sha256": "1" * 64, "dimension": 8},
        )
        self.calls: list[tuple[Path, ...]] = []
        self.fail_if_called = fail_if_called

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        if self.fail_if_called:
            raise AssertionError("DINO cache miss during progress resume")
        self.calls.append(tuple(Path(path) for path in paths))
        rows = []
        for path in paths:
            value = np.zeros(8, dtype=np.float32)
            if path.name == "goal_1.jpg":
                # Frame 3 wins at midpoint; causal t0 cannot see it and picks 1.
                value[3] = 1.0
                value[1] = 0.5
            elif path.name == "goal_2.jpg":
                value[1] = 1.0
            else:
                value[int(path.stem)] = 1.0
            rows.append(value)
        return np.stack(rows)


class FakeRenderer:
    """Pinned renderer double returning one-metre metric depth."""

    def __init__(
        self,
        *,
        fail_if_called: bool = False,
        mutate_glb_after_first_call: bool = False,
    ) -> None:
        self.identity = make_provider_identity(
            RENDERER_IDENTITY_SCHEMA,
            "fixture_habitat_depth_renderer",
            {"habitat_version": "fixture", "sensor_model": "depth_m"},
        )
        self.calls: list[dict[str, Any]] = []
        self.fail_if_called = fail_if_called
        self.mutate_glb_after_first_call = mutate_glb_after_first_call

    def render_depth(self, **request: Any) -> np.ndarray:
        if self.fail_if_called:
            raise AssertionError("geometry cache miss during progress resume")
        captured = dict(request)
        captured["camera_position_habitat"] = np.asarray(
            request["camera_position_habitat"], dtype=np.float64
        ).copy()
        captured["intrinsic"] = np.asarray(
            request["intrinsic"], dtype=np.float64
        ).copy()
        self.calls.append(captured)
        if self.mutate_glb_after_first_call and len(self.calls) == 1:
            glb_path = Path(request["glb_path"])
            glb_path.write_bytes(glb_path.read_bytes() + b"mutated-during-render")
        return np.ones((request["height"], request["width"]), dtype=np.float32)

    def close(self) -> None:
        return None


class FakeStreamingDINO(FakeDINO):
    """Streaming double that observes atomic publication between yields."""

    def __init__(self, progress: Path) -> None:
        super().__init__()
        self.progress = progress

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        raise AssertionError("streaming provider fell back to monolithic embed")

    def embed_chunks(
        self, path_chunks: Sequence[Sequence[Path]]
    ) -> Sequence[np.ndarray]:
        for index, paths in enumerate(path_chunks):
            if index:
                assert len(list((self.progress / "embeddings").iterdir())) == index
            yield FakeDINO.embed(self, paths)


class TeacherFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.episodes = root / "episodes"
        self.environments = root / "environments"
        self.episodes.mkdir()
        self.environments.mkdir()
        self.environment_path = self.environments / f"{SCENE}.glb"
        self.environment_path.write_bytes(b"pinned-fixture-glb")
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self._write_episode(SOURCE_EPISODE, factual=True)
        self._write_episode(PARTNER_EPISODE, factual=False)
        self.manifest = self._manifest()

    @staticmethod
    def _intrinsic() -> list[list[float]]:
        return [
            [4.0, 0.0, 3.5],
            [0.0, 4.0, 3.5],
            [0.0, 0.0, 1.0],
        ]

    @staticmethod
    def _camera_to_world() -> list[list[float]]:
        # This is exactly M_W @ R_yaw(0), with a one-metre Habitat camera
        # height converted to stored Z-up coordinates.
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def episode_path(self, episode: str) -> Path:
        return self.episodes / SCENE / episode

    def rgb_path(self, episode: str, frame: int) -> Path:
        return (
            self.episode_path(episode)
            / "videos/chunk-000/observation.images.rgb"
            / f"{frame}.jpg"
        )

    def depth_path(self, episode: str, frame: int) -> Path:
        return (
            self.episode_path(episode)
            / "videos/chunk-000/observation.images.depth"
            / f"{frame}.png"
        )

    def metadata_path(self, episode: str) -> Path:
        return self.episode_path(episode) / "meta/gen_meta.json"

    def goal_path(self, episode: str, role: str) -> Path:
        return self.episode_path(episode) / (
            "goal_1.jpg" if role == "B" else "goal_2.jpg"
        )

    def _write_episode(self, episode: str, *, factual: bool) -> None:
        root = self.episode_path(episode)
        rgb_root = root / "videos/chunk-000/observation.images.rgb"
        depth_root = root / "videos/chunk-000/observation.images.depth"
        parquet_root = root / "data/chunk-000"
        metadata_root = root / "meta"
        for path in (rgb_root, depth_root, parquet_root, metadata_root):
            path.mkdir(parents=True, exist_ok=True)

        for frame in range(N_FRAMES):
            rgb = np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                (frame * 29 + (0 if factual else 7)) % 255,
                dtype=np.uint8,
            )
            Image.fromarray(rgb, mode="RGB").save(
                self.rgb_path(episode, frame), format="JPEG"
            )
            depth = np.full((IMAGE_SIZE, IMAGE_SIZE), 10000, dtype=np.uint16)
            Image.fromarray(depth).save(self.depth_path(episode, frame), format="PNG")

        for role, color in (("B", 83), ("C", 151)):
            goal = np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                color + (0 if factual else 11),
                dtype=np.uint8,
            )
            Image.fromarray(goal, mode="RGB").save(
                self.goal_path(episode, role), format="JPEG"
            )

        b_curve = [0.11, 0.82] if factual else [0.01] * 2
        c_curve = [0.10, 0.93, 0.20, 0.30, 0.40] if factual else [0.02] * 5
        goal_position = [0.0, 0.0, 0.0] if factual else [2.0, -3.0, 0.0]
        goal_yaw = 0.0 if factual else 0.25
        metadata = {
            "scene": f"{SCENE}.glb",
            "ep_idx": int(episode.removeprefix("episode_")),
            "n_frames": N_FRAMES,
            "n_legs": 3,
            "switches": [2, 5],
            "camera_height_m": 1.0,
            "frame_convention": (
                "positions+parquet in data(Zup,M_W); yaw_habitat in render frame"
            ),
            "goals": [
                {
                    "name": "B",
                    "kind": "novel",
                    "pos": goal_position,
                    "yaw_habitat": goal_yaw,
                    "covis_curve": b_curve,
                },
                {
                    "name": "C",
                    "kind": "revisit",
                    "pos": goal_position,
                    "yaw_habitat": goal_yaw,
                    "covis_curve": c_curve,
                },
            ],
        }
        self.metadata[episode] = metadata
        self.metadata_path(episode).write_bytes(canonical_json_bytes(metadata))

        intrinsic = self._intrinsic()
        action = self._camera_to_world()
        extrinsic = np.eye(4, dtype=np.float64).tolist()
        rows = [
            {
                "index": frame,
                "observation.camera_intrinsic": intrinsic,
                "observation.camera_extrinsic": extrinsic,
                "action": action,
            }
            for frame in range(N_FRAMES)
        ]
        self.rows[episode] = rows
        pq.write_table(
            pa.Table.from_pylist(rows),
            root / "data/chunk-000/episode_000000.parquet",
        )

    def _prefix(self, episode: str, decision: int) -> dict[str, Any]:
        modalities: dict[str, dict[str, str]] = {}
        for modality, suffix in (("rgb", ".jpg"), ("depth", ".png")):
            records = []
            for frame in range(decision):
                path = (
                    self.rgb_path(episode, frame)
                    if modality == "rgb"
                    else self.depth_path(episode, frame)
                )
                record = _file_record(path, self.episodes)
                assert record["path"].endswith(suffix)
                records.append(record)
            modalities[modality] = {
                "path_sequence_sha256": _sequence_sha(
                    [record["path"] for record in records]
                ),
                "content_sequence_sha256": _sequence_sha(
                    [
                        {
                            "path": record["path"],
                            "bytes": record["bytes"],
                            "content_sha256": record["content_sha256"],
                        }
                        for record in records
                    ]
                ),
            }
        rows = self.rows[episode][:decision]
        parquet_sha = _sequence_sha(rows)
        payload = {
            "frame_count": decision,
            "rgb": modalities["rgb"],
            "depth": modalities["depth"],
            "parquet_rows_sha256": parquet_sha,
        }
        return {
            "exclusive_end_frame": decision,
            "frame_count": decision,
            "modalities": modalities,
            "parquet_columns": list(PARQUET_PREFIX_COLUMNS),
            "parquet_row_count": decision,
            "parquet_rows_sha256": parquet_sha,
            "causal_prefix_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }

    def _episode_record(self, episode: str) -> dict[str, Any]:
        root = self.episode_path(episode)
        return {
            "episode": episode,
            "n_frames": N_FRAMES,
            "switches": [2, 5],
            "goal_b_midpoint_frame": 4,
            "metadata": _file_record(self.metadata_path(episode), self.episodes),
            "parquet": _file_record(
                root / "data/chunk-000/episode_000000.parquet", self.episodes
            ),
            "goal_b": _file_record(self.goal_path(episode, "B"), self.episodes),
            "goal_c": _file_record(self.goal_path(episode, "C"), self.episodes),
        }

    def _sample(
        self,
        *,
        state_name: str,
        decision: int,
        role: str,
        variant: str,
    ) -> dict[str, Any]:
        goal_episode = SOURCE_EPISODE if variant == "factual" else PARTNER_EPISODE
        goal = _file_record(self.goal_path(goal_episode, role), self.episodes)
        return {
            "sample_id": f"train/{SCENE}/{SOURCE_EPISODE}/{state_name}/{variant}",
            "split_role": "train",
            "scene": SCENE,
            "state_source": "expert",
            "source_episode": SOURCE_EPISODE,
            "source_episode_id": f"{SCENE}/{SOURCE_EPISODE}",
            "goal_episode": goal_episode,
            "goal_source_episode_id": f"{SCENE}/{goal_episode}",
            "goal_variant": variant,
            "goal_role": role,
            "state_name": state_name,
            "decision_frame": decision,
            "state_frame": _file_record(
                self.rgb_path(SOURCE_EPISODE, decision - 1), self.episodes
            ),
            "causal_prefix": self._prefix(SOURCE_EPISODE, decision),
            "navdp_fifo": {},
            "goal": goal,
        }

    def _manifest(self) -> dict[str, Any]:
        episode_records = [
            self._episode_record(SOURCE_EPISODE),
            self._episode_record(PARTNER_EPISODE),
        ]
        samples = []
        for state_name, decision, role in (
            ("goal_b_t0", 2, "B"),
            ("goal_b_midpoint_t1", 4, "B"),
            ("goal_c_t0", 5, "C"),
        ):
            for variant in ("factual", "counterfactual"):
                samples.append(
                    self._sample(
                        state_name=state_name,
                        decision=decision,
                        role=role,
                        variant=variant,
                    )
                )
        bindings = []
        for episode, record in zip(
            (SOURCE_EPISODE, PARTNER_EPISODE), episode_records, strict=True
        ):
            bindings.append(
                {
                    "split_role": "train",
                    "scene": SCENE,
                    "episode": episode,
                    "episode_id": f"{SCENE}/{episode}",
                    "camera_height_m": 1.0,
                    "value_source": "episode_metadata:meta/gen_meta.json#camera_height_m",
                    "metadata_content_sha256": record["metadata"]["content_sha256"],
                }
            )
        return {
            "schema_version": MANIFEST_SCHEMA,
            "purpose": "pure fixture formal multistage manifest",
            "input_roots": {
                "episode_root": str(self.episodes.resolve()),
                "environment_root": str(self.environments.resolve()),
            },
            "provenance": {"camera_height_bindings": bindings},
            "scenes": [
                {
                    "scene": SCENE,
                    "split_role": "train",
                    "environment": _file_record(
                        self.environment_path, self.environments
                    ),
                    "selected_episodes": episode_records,
                }
            ],
            "samples": samples,
            "summary": {"sample_count": len(samples)},
        }

    def build(
        self,
        *,
        manifest: Mapping[str, Any] | None = None,
        dino: FakeDINO | None = None,
        renderer: FakeRenderer | None = None,
        progress_directory: Path | None = None,
        embedding_chunk_size: int = 256,
    ) -> tuple[dict[str, Any], FakeDINO, FakeRenderer]:
        source = self.manifest if manifest is None else manifest
        provider = FakeDINO() if dino is None else dino
        depth_renderer = FakeRenderer() if renderer is None else renderer
        artifact = build_teacher_artifact(
            manifest=source,
            manifest_sha256=sha256_bytes(canonical_json_bytes(source)),
            embedding_provider=provider,
            renderer=depth_renderer,
            config=TeacherConfig(
                top_k=1,
                temporal_nms_radius=0,
                backprojection_stride=1,
            ),
            expected_sample_count=6,
            progress_directory=progress_directory,
            embedding_chunk_size=embedding_chunk_size,
        )
        return artifact, provider, depth_renderer

    def build_embedding_bundle(
        self,
        *,
        dino: FakeDINO | None = None,
        chunk_size: int = 2,
    ) -> tuple[Path, str, FakeDINO]:
        provider = FakeDINO() if dino is None else dino
        progress = self.root / "embedding_progress"
        output = self.root / "embedding_bundle"
        result = build_dino_embedding_bundle(
            manifest=self.manifest,
            manifest_sha256=sha256_bytes(canonical_json_bytes(self.manifest)),
            embedding_provider=provider,
            progress_directory=progress,
            output_directory=output,
            expected_sample_count=6,
            embedding_chunk_size=chunk_size,
        )
        return output, str(result["receipt_sha256"]), provider


@pytest.fixture
def teacher_fixture(tmp_path: Path) -> TeacherFixture:
    return TeacherFixture(tmp_path)


def _record_index(
    artifact: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (record["state_name"], record["goal_variant"]): record
        for record in artifact["records"]
    }


def test_temporal_nms_is_stable_and_suppresses_adjacent_frames() -> None:
    assert temporal_nms_shortlist(
        [0, 1, 2, 5],
        [0.9, 0.9, 0.8, 0.7],
        top_k=3,
        radius=1,
    ) == [(0, 0.9), (2, 0.8), (5, 0.7)]


def test_exact_cover_and_no_future_candidate_universe(
    teacher_fixture: TeacherFixture,
) -> None:
    artifact, dino, _renderer = teacher_fixture.build()
    expected_ids = [
        sample["sample_id"] for sample in teacher_fixture.manifest["samples"]
    ]
    assert [record["sample_id"] for record in artifact["records"]] == expected_ids
    assert artifact["exact_cover"] == {
        "manifest_sample_count": 6,
        "output_sample_count": 6,
        "sample_ids_equal_in_manifest_order": True,
        "sample_id_sequence_sha256": sha256_bytes(canonical_json_bytes(expected_ids)),
    }
    assert artifact["summary"]["samples"] == 6
    assert artifact["summary"]["candidates"] == 6

    for record in artifact["records"]:
        decision = record["decision_frame"]
        assert record["selection"]["universe_frame_count"] == decision
        assert record["no_future_source_observation"] is True
        assert all(
            candidate["no_future"] is True
            and 0 <= candidate["candidate_frame"] < decision
            for candidate in record["candidates"]
        )

    indexed = _record_index(artifact)
    # Frame 3 is the strongest B match but is forbidden at the t0 decision.
    assert indexed[("goal_b_t0", "factual")]["candidates"][0]["candidate_frame"] == 1
    assert (
        indexed[("goal_b_midpoint_t1", "factual")]["candidates"][0]["candidate_frame"]
        == 3
    )

    assert len(dino.calls) == 1
    seen = {path.resolve() for path in dino.calls[0]}
    # Physical frame 5 exists, but no sample may consume it because the latest
    # decision is exclusive_end=5.
    assert teacher_fixture.rgb_path(SOURCE_EPISODE, 5).resolve() not in seen


def test_label_authority_is_metadata_only_when_episode_local_and_in_range(
    teacher_fixture: TeacherFixture,
) -> None:
    artifact, _dino, renderer = teacher_fixture.build()
    indexed = _record_index(artifact)

    b_t0 = indexed[("goal_b_t0", "factual")]["candidates"][0]
    assert b_t0["candidate_frame"] == 1
    assert b_t0["label_source"] == "metadata_covis_curve"
    assert b_t0["covisibility"] == pytest.approx(0.82)
    assert b_t0["rendered_goal_depth"] is None

    c_factual = indexed[("goal_c_t0", "factual")]["candidates"][0]
    assert c_factual["candidate_frame"] == 1
    assert c_factual["label_source"] == "metadata_covis_curve"
    assert c_factual["covisibility"] == pytest.approx(0.93)
    assert c_factual["rendered_goal_depth"] is None

    # Goal-B's stored curve ends at frame 1; midpoint frame 3 must therefore
    # fall back to freshly rendered, occlusion-aware geometry.
    b_midpoint = indexed[("goal_b_midpoint_t1", "factual")]["candidates"][0]
    assert b_midpoint["candidate_frame"] == 3
    assert b_midpoint["label_source"] == "rendered_goal_depth_reprojection"
    assert b_midpoint["rendered_goal_depth"] is not None

    counterfactuals = [
        record["candidates"][0]
        for record in artifact["records"]
        if record["goal_variant"] == "counterfactual"
    ]
    assert len(counterfactuals) == 3
    for candidate in counterfactuals:
        assert candidate["label_source"] == "rendered_goal_depth_reprojection"
        assert candidate["rendered_goal_depth"] is not None
        assert "covis_curve_sha256" not in candidate["label_input"]
        assert "metadata_content_sha256" not in candidate["label_input"]
        # Partner metadata contains only sentinel 0.01/0.02 scores.  The
        # geometric label must therefore be observably independent of it.
        assert candidate["covisibility"] not in (0.01, 0.02)

    assert artifact["summary"]["label_sources"] == {
        "metadata_covis_curve": 2,
        "rendered_goal_depth_reprojection": 4,
    }
    assert renderer.calls
    assert any(
        np.allclose(call["camera_position_habitat"], [0.0, 1.0, 0.0])
        and call["yaw_habitat"] == 0.0
        for call in renderer.calls
    )
    # Partner metadata stores [2,-3,0] in data Z-up.  The renderer must use the
    # cross-episode goal-source pose [2,0,3] in Habitat plus camera height.
    assert any(
        np.allclose(call["camera_position_habitat"], [2.0, 1.0, 3.0])
        and call["yaw_habitat"] == 0.25
        for call in renderer.calls
    )
    assert renderer.calls[0]["glb_path"] == teacher_fixture.environment_path


@pytest.mark.parametrize("sample_index", [0, 1])
def test_declared_prefix_tamper_fails_closed(
    teacher_fixture: TeacherFixture,
    sample_index: int,
) -> None:
    manifest = copy.deepcopy(teacher_fixture.manifest)
    manifest["samples"][sample_index]["causal_prefix"]["frame_count"] += 1
    with pytest.raises(
        CausalTeacherError,
        match="causal prefix (content changed|conflicts within state pair)",
    ):
        teacher_fixture.build(manifest=manifest)


@pytest.mark.parametrize(
    "target", ["goal", "metadata", "candidate_rgb", "candidate_depth"]
)
def test_pinned_content_tamper_fails_closed(
    teacher_fixture: TeacherFixture,
    target: str,
) -> None:
    targets = {
        "goal": teacher_fixture.goal_path(SOURCE_EPISODE, "C"),
        "metadata": teacher_fixture.metadata_path(PARTNER_EPISODE),
        "candidate_rgb": teacher_fixture.rgb_path(SOURCE_EPISODE, 1),
        "candidate_depth": teacher_fixture.depth_path(SOURCE_EPISODE, 3),
    }
    path = targets[target]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(CausalTeacherError):
        teacher_fixture.build()


@pytest.mark.parametrize("provider_kind", ["dino", "renderer"])
def test_provider_identity_tamper_fails_closed(
    teacher_fixture: TeacherFixture,
    provider_kind: str,
) -> None:
    dino = FakeDINO()
    renderer = FakeRenderer()
    provider = dino if provider_kind == "dino" else renderer
    provider.identity["components"]["tampered_after_fingerprint"] = True
    with pytest.raises(CausalTeacherError, match="identity fingerprint mismatch"):
        teacher_fixture.build(dino=dino, renderer=renderer)


def test_provider_identities_are_bound_into_artifact(
    teacher_fixture: TeacherFixture,
) -> None:
    artifact, dino, renderer = teacher_fixture.build()
    assert artifact["dino_provider"] == dino.identity
    assert artifact["goal_depth_renderer"] == renderer.identity
    assert artifact["deployment_approved"] is False


def test_glb_change_between_goal_renders_fails_closed(
    teacher_fixture: TeacherFixture,
) -> None:
    with pytest.raises(
        CausalTeacherError, match="scene GLB changed between goal renders"
    ):
        teacher_fixture.build(renderer=FakeRenderer(mutate_glb_after_first_call=True))


def test_progress_resume_reuses_dino_and_sample_geometry_shards(
    teacher_fixture: TeacherFixture,
) -> None:
    progress = teacher_fixture.root / "teacher_progress"
    first, first_dino, first_renderer = teacher_fixture.build(
        progress_directory=progress,
        embedding_chunk_size=2,
    )
    assert len(first_dino.calls) == 1
    assert first_renderer.calls

    resumed, resumed_dino, resumed_renderer = teacher_fixture.build(
        dino=FakeDINO(fail_if_called=True),
        renderer=FakeRenderer(fail_if_called=True),
        progress_directory=progress,
        embedding_chunk_size=2,
    )
    assert canonical_json_bytes(resumed) == canonical_json_bytes(first)
    assert resumed_dino.calls == []
    assert resumed_renderer.calls == []

    # Simulate interruption after all embeddings but before the first factual
    # sample is committed.  Rebuilding that metadata-only row must not rerun
    # DINO or Habitat and must recover the exact original artifact.
    sample_shards = sorted((progress / "samples").iterdir())
    assert len(sample_shards) == 6
    shutil.rmtree(sample_shards[0])
    recovered, _dino, _renderer = teacher_fixture.build(
        dino=FakeDINO(fail_if_called=True),
        renderer=FakeRenderer(fail_if_called=True),
        progress_directory=progress,
        embedding_chunk_size=2,
    )
    assert canonical_json_bytes(recovered) == canonical_json_bytes(first)

    embedding_shard = sorted((progress / "embeddings").iterdir())[0]
    array_path = embedding_shard / "embeddings.npy"
    array_path.write_bytes(array_path.read_bytes() + b"tamper")
    with pytest.raises(CausalTeacherError, match="embedding shard content changed"):
        teacher_fixture.build(
            dino=FakeDINO(fail_if_called=True),
            renderer=FakeRenderer(fail_if_called=True),
            progress_directory=progress,
            embedding_chunk_size=2,
        )


def test_progress_run_signature_drift_fails_closed(
    teacher_fixture: TeacherFixture,
) -> None:
    progress = teacher_fixture.root / "teacher_progress"
    teacher_fixture.build(progress_directory=progress, embedding_chunk_size=2)
    with pytest.raises(CausalTeacherError, match="progress run signature drifted"):
        teacher_fixture.build(
            progress_directory=progress,
            embedding_chunk_size=3,
        )


def test_canonical_bundle_audit_sidecars_and_fail_closed_resume(
    teacher_fixture: TeacherFixture,
) -> None:
    artifact, _dino, _renderer = teacher_fixture.build()
    expected = teacher_bundle_payloads(artifact)
    output = teacher_fixture.root / "teacher_bundle"

    written = write_teacher_bundle(artifact, output)
    assert written["status"] == "written"
    assert {path.name for path in output.iterdir()} == BUNDLE_FILES
    assert {name: (output / name).read_bytes() for name in BUNDLE_FILES} == expected
    assert (output / ARTIFACT_NAME).read_bytes() == canonical_json_bytes(artifact)
    for name in (ARTIFACT_NAME, CSV_NAME, AUDIT_NAME):
        assert (output / f"{name}.sha256").read_bytes() == (
            f"{sha256_bytes((output / name).read_bytes())}  {name}\n"
        ).encode("ascii")

    audit = json.loads((output / AUDIT_NAME).read_text(encoding="utf-8"))
    assert audit["counts"] == {
        "samples": 6,
        "candidates": 6,
        "counterfactual_candidates": 3,
        "label_sources": {
            "metadata_covis_curve": 2,
            "rendered_goal_depth_reprojection": 4,
        },
    }
    assert all(audit["invariants"].values())
    assert audit["authority"]["dino_or_ransac_self_report_used_as_label"] is False
    assert audit["deployment_approved"] is False
    assert len((output / CSV_NAME).read_text(encoding="utf-8").splitlines()) == 7

    csv_rows = list(
        csv.DictReader(io.StringIO((output / CSV_NAME).read_text(encoding="utf-8")))
    )
    assert len(csv_rows) == 6
    phase_b_required = {
        "session_id",
        "scene",
        "episode",
        "kind",
        "query_path",
        "candidate_path",
        "candidate_frame",
        "dino_cosine",
        "teacher_covis",
        "label",
        "causal_manifest_sample_id",
    }
    assert phase_b_required <= set(csv_rows[0])
    for row in csv_rows:
        assert row["session_id"] == row["sample_id"]
        assert row["causal_manifest_sample_id"] == row["sample_id"]
        assert row["episode"] == row["source_episode"]
        assert row["kind"] == _phase_b_kind(row["split_role"])
        assert float(row["teacher_covis"]) == float(row["covisibility"])
        assert Path(row["query_path"]).is_absolute()
        assert Path(row["query_path"]).is_file()
        assert Path(row["candidate_path"]).is_absolute()
        assert Path(row["candidate_path"]).is_file()
        assert int(row["candidate_frame"]) < int(row["decision_frame"])
        assert row["no_future"] == "true"
    assert audit["csv"]["phase_b_contract"]["exact_candidate_cover"] is True
    assert audit["invariants"]["phase_b_csv_exact_candidate_cover"] is True

    resumed = write_teacher_bundle(artifact, output, resume=True)
    assert resumed["status"] == "resumed"
    with pytest.raises(CausalTeacherError, match="already exists"):
        write_teacher_bundle(artifact, output)

    (output / CSV_NAME).write_bytes((output / CSV_NAME).read_bytes() + b"drift")
    with pytest.raises(CausalTeacherError, match="content drifted"):
        write_teacher_bundle(artifact, output, resume=True)

    partial = teacher_fixture.root / "partial_bundle"
    partial.mkdir()
    (partial / ARTIFACT_NAME).write_bytes(expected[ARTIFACT_NAME])
    with pytest.raises(CausalTeacherError, match="file set changed"):
        write_teacher_bundle(artifact, partial, resume=True)


def _teacher_semantics(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Runtime-neutral view used to compare live and staged DINO execution."""

    result = []
    for record in artifact["records"]:
        result.append(
            {
                "sample_id": record["sample_id"],
                "candidate_domain": record["candidate_frame_domain"],
                "selection": {
                    key: record["selection"][key]
                    for key in (
                        "universe_frame_count",
                        "universe_scores_sha256",
                        "selected_frame_indices",
                    )
                },
                "candidates": [
                    {
                        key: candidate[key]
                        for key in (
                            "candidate_frame",
                            "dino_cosine",
                            "label_source",
                            "covisibility",
                            "label",
                            "label_input_sha256",
                        )
                    }
                    for candidate in record["candidates"]
                ],
            }
        )
    return result


def test_two_stage_bundle_matches_live_teacher_semantics_and_never_imports_torch(
    teacher_fixture: TeacherFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, _dino, _renderer = teacher_fixture.build()
    bundle, receipt_sha, stage_a_dino = teacher_fixture.build_embedding_bundle()
    assert len(stage_a_dino.calls) == 1
    assert (bundle / EMBEDDING_RECEIPT_NAME).is_file()

    real_import = builtins.__import__

    def reject_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Stage B attempted to import Torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    cached = PinnedDINOEmbeddingBundleProvider(
        bundle_directory=bundle,
        expected_receipt_sha256=receipt_sha,
        expected_manifest_sha256=sha256_bytes(
            canonical_json_bytes(teacher_fixture.manifest)
        ),
        episode_root=teacher_fixture.episodes,
    )
    staged = build_teacher_artifact(
        manifest=teacher_fixture.manifest,
        manifest_sha256=sha256_bytes(canonical_json_bytes(teacher_fixture.manifest)),
        embedding_provider=cached,
        renderer=FakeRenderer(),
        config=TeacherConfig(
            top_k=1,
            temporal_nms_radius=0,
            backprojection_stride=1,
        ),
        expected_sample_count=6,
        progress_directory=teacher_fixture.root / "assembly_progress",
    )
    assert _teacher_semantics(staged) == _teacher_semantics(live)
    assert (
        staged["embedding_invocation"]["provider_model_loads_per_process_run_max"] == 0
    )
    assert staged["embedding_invocation"]["embedding_source"] == cached.bundle_binding
    assert not any(
        (teacher_fixture.root / "assembly_progress" / "embeddings").iterdir()
    )


def test_stage_entrypoints_do_not_reference_the_other_runtime() -> None:
    root = Path(__file__).resolve().parent
    stage_a = (root / "build_manifest_causal_dino_embeddings.py").read_text(
        encoding="utf-8"
    )
    stage_b = (root / "assemble_manifest_causal_covisibility_teacher.py").read_text(
        encoding="utf-8"
    )
    assert "PinnedHabitatGoalDepthRenderer" not in stage_a
    assert "ExactLingBotDINOProvider" not in stage_b
    assert "import torch" not in stage_b


@pytest.mark.parametrize("target", ["receipt", "shard", "rgb"])
def test_signed_embedding_bundle_tamper_or_cache_miss_fails_closed(
    teacher_fixture: TeacherFixture,
    target: str,
) -> None:
    bundle, receipt_sha, _dino = teacher_fixture.build_embedding_bundle()
    manifest_sha = sha256_bytes(canonical_json_bytes(teacher_fixture.manifest))
    if target == "receipt":
        receipt = bundle / EMBEDDING_RECEIPT_NAME
        receipt.write_bytes(receipt.read_bytes() + b"tamper")
        with pytest.raises(
            CausalTeacherError, match="receipt content or sidecar changed"
        ):
            PinnedDINOEmbeddingBundleProvider(
                bundle_directory=bundle,
                expected_receipt_sha256=receipt_sha,
                expected_manifest_sha256=manifest_sha,
                episode_root=teacher_fixture.episodes,
            )
        return

    provider = PinnedDINOEmbeddingBundleProvider(
        bundle_directory=bundle,
        expected_receipt_sha256=receipt_sha,
        expected_manifest_sha256=manifest_sha,
        episode_root=teacher_fixture.episodes,
    )
    receipt = json.loads((bundle / EMBEDDING_RECEIPT_NAME).read_text(encoding="utf-8"))
    paths = [
        teacher_fixture.episodes / record["path"] for record in receipt["input_records"]
    ]
    with pytest.raises(
        CausalTeacherError, match="not the exact Stage-A input sequence"
    ):
        provider.embed(paths[:-1])
    if target == "shard":
        shard = sorted((bundle / "shards").iterdir())[0] / "embeddings.npy"
        shard.write_bytes(shard.read_bytes() + b"tamper")
        expected = "embedding shard changed after provider construction"
    else:
        paths[0].write_bytes(paths[0].read_bytes() + b"tamper")
        expected = "signed embedding physical RGB input changed"
    with pytest.raises(CausalTeacherError, match=expected):
        provider.embed(paths)


@pytest.mark.parametrize("field", ["action", "observation.camera_extrinsic"])
def test_non_rigid_parquet_pose_fails_before_geometry(
    teacher_fixture: TeacherFixture,
    field: str,
) -> None:
    rows = teacher_fixture.rows[SOURCE_EPISODE]
    rows[0][field][0][0] = 2.0
    parquet_path = (
        teacher_fixture.episode_path(SOURCE_EPISODE)
        / "data/chunk-000/episode_000000.parquet"
    )
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    manifest = teacher_fixture._manifest()
    with pytest.raises(
        CausalTeacherError, match=r"rotation is not a proper SO\(3\) matrix"
    ):
        teacher_fixture.build(manifest=manifest)


def test_nonzero_yaw_goal_pose_uses_one_consistent_habitat_and_data_rotation() -> None:
    yaw = 0.37
    camera_habitat, camera_to_world_data = _goal_camera_pose(
        {
            "position_data_zup_m": [2.0, -3.0, 0.0],
            "yaw_habitat_rad": yaw,
        },
        1.0,
    )
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    expected_habitat_rotation = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    assert np.allclose(camera_habitat, [2.0, 1.0, 3.0])
    assert np.allclose(camera_to_world_data[:3, :3], M_W @ expected_habitat_rotation)
    assert np.allclose(camera_to_world_data[:3, 3], M_W @ camera_habitat)
    assert not np.allclose(camera_to_world_data[:3, :3], M_W)
    assert np.allclose(
        camera_to_world_data[:3, :3].T @ camera_to_world_data[:3, :3],
        np.eye(3),
    )
    assert np.linalg.det(camera_to_world_data[:3, :3]) == pytest.approx(1.0)


def test_stage_a_streams_and_atomically_publishes_each_chunk(
    teacher_fixture: TeacherFixture,
) -> None:
    progress = teacher_fixture.root / "streaming_progress"
    output = teacher_fixture.root / "streaming_bundle"
    provider = FakeStreamingDINO(progress)
    result = build_dino_embedding_bundle(
        manifest=teacher_fixture.manifest,
        manifest_sha256=sha256_bytes(canonical_json_bytes(teacher_fixture.manifest)),
        embedding_provider=provider,
        progress_directory=progress,
        output_directory=output,
        expected_sample_count=6,
        embedding_chunk_size=2,
    )
    assert result["status"] == "written"
    assert len(provider.calls) > 1
    assert len(list((progress / "embeddings").iterdir())) == len(provider.calls)
    assert len(list((output / "shards").iterdir())) == len(provider.calls)


def test_stage_a_resume_requires_flag_and_reuses_all_signed_shards(
    teacher_fixture: TeacherFixture,
) -> None:
    bundle, receipt_sha, _provider = teacher_fixture.build_embedding_bundle()
    arguments = {
        "manifest": teacher_fixture.manifest,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(teacher_fixture.manifest)),
        "embedding_provider": FakeDINO(fail_if_called=True),
        "progress_directory": teacher_fixture.root / "embedding_progress",
        "output_directory": bundle,
        "expected_sample_count": 6,
        "embedding_chunk_size": 2,
    }
    with pytest.raises(CausalTeacherError, match="already exists without --resume"):
        build_dino_embedding_bundle(**arguments)
    result = build_dino_embedding_bundle(**arguments, resume=True)
    assert result["status"] == "resumed"
    assert result["receipt_sha256"] == receipt_sha


def test_pinned_runtime_dependency_drift_fails_closed(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.py"
    dependency.write_text("VERSION = 1\n", encoding="utf-8")
    records = {
        "fixture_dependency": _physical_runtime_record(dependency, "fixture dependency")
    }
    _verify_static_runtime_records(records, label="fixture runtime")
    dependency.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(CausalTeacherError, match="runtime dependency changed"):
        _verify_static_runtime_records(records, label="fixture runtime")


def test_stage_a_rejects_a_second_writer(tmp_path: Path) -> None:
    progress = tmp_path / "embedding_progress"
    with _exclusive_stage_writer(progress):
        with pytest.raises(CausalTeacherError, match="another bundle writer"):
            with _exclusive_stage_writer(progress):
                raise AssertionError("second writer unexpectedly entered")


def test_scene_work_is_grouped_but_retains_original_manifest_indices() -> None:
    contexts = [
        SimpleNamespace(sample={"scene": scene})
        for scene in ("scene_b", "scene_a", "scene_b", "scene_a")
    ]
    work = _scene_grouped_work_items(contexts)  # type: ignore[arg-type]
    assert [index for index, _context in work] == [1, 3, 0, 2]
    restored = [None] * len(work)
    for index, context in work:
        restored[index] = context.sample["scene"]
    assert restored == ["scene_b", "scene_a", "scene_b", "scene_a"]


def test_candidate_oracle_summary_is_partitioned_and_audit_bound(
    teacher_fixture: TeacherFixture,
) -> None:
    artifact, _dino, _renderer = teacher_fixture.build()
    oracle = artifact["summary"]["candidate_oracle"]
    overall = oracle["groups"]["overall"]
    assert overall["sessions"] == 6
    assert (
        overall["session_has_positive"]
        + overall["strict_shortlist_no_match"]
        + overall["shortlist_ambiguous"]
        == 6
    )
    assert set(overall["shortlist_conditional_positive_recall_at_k"]) == {
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
    }
    assert {
        "joint/train/B/factual",
        "joint/train/B/counterfactual",
        "joint/train/C/factual",
        "joint/train/C/counterfactual",
    } <= set(oracle["groups"])
    audit = json.loads(teacher_bundle_payloads(artifact)[AUDIT_NAME])
    assert audit["candidate_oracle"] == oracle


def test_phase_b_collector_consumes_teacher_csv_without_adapter(
    teacher_fixture: TeacherFixture,
) -> None:
    import pandas as pd

    from MemNavData.diag_lingbot_goal_loop_closure import (
        select_deployment_seeds,
    )

    artifact, _dino, _renderer = teacher_fixture.build()
    frame = pd.read_csv(io.BytesIO(teacher_bundle_payloads(artifact)[CSV_NAME]))
    seeds = select_deployment_seeds(
        frame,
        kind=_phase_b_kind("train"),
        sessions=(),
        max_sessions=0,
        top_k=1,
        minimum_gap=0,
        positive_threshold=0.5,
        negative_threshold=0.1,
        minimum_anchor=0,
    )
    assert len(seeds) == 6
    assert all(seed.session_id == seed.causal_manifest_sample_id for seed in seeds)
    assert all(seed.query_path.is_file() for seed in seeds)
    assert all(seed.candidate_path.is_file() for seed in seeds)
