"""Audited HTTP proxy for frozen alternate navigation controllers.

The baseline servers bundled by NavDP predate this project's causal and
provenance contracts.  This proxy leaves their model code unchanged while it
adds a checkpoint/source receipt, rejects role/oracle fields, validates CEC's
fixed-radius PointGoal, and checks that returned trajectories are finite.

It is an evaluation wrapper, not a new planner and not a fallback router.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from flask import Flask, jsonify, request
import numpy as np
import requests

from controller_portability_contract import (
    CEC_BEARING_EXECUTOR,
    CEC_PROOF_HYBRID,
    CONTROLLER_PORTABILITY_SCHEMA_VERSION,
    NATIVE_IMAGEGOAL,
    ComparisonPlan,
    controller_spec,
    fixed_bearing_payload,
    validate_comparison_plan,
)


PROXY_SCHEMA = "cec_controller_portability_proxy_v2"
FORBIDDEN_RUNTIME_FIELDS = frozenset({
    "role",
    "goal_role",
    "query_role",
    "is_revisit",
    "is_novel",
    "oracle_pose",
    "gt_pose",
    "ground_truth_pose",
    "habitat_pose",
})
EXPECTED_ALGO = {
    "vint": "vint",
    "gnm": "gnm",
    "nomad": "nomad",
    "iplanner": "iplanner",
    "viplanner": "viplanner",
}
REQUIRED_CHECKPOINT_LABELS = {
    "vint": frozenset({"vint"}),
    "gnm": frozenset({"gnm"}),
    "nomad": frozenset({"nomad"}),
    "iplanner": frozenset({"iplanner"}),
    "viplanner": frozenset({"planner", "mask2former"}),
}
# Controllers whose native server keeps a short causal RGB context (a
# memory_queue of the last few frames) that must be shadow-advanced via
# /observation_step whenever another controller owns the current action.
OBSERVATION_STEP_CONTROLLERS = frozenset({"vint", "gnm", "nomad"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_tree(path: Path) -> str:
    """Hash source contents and relative names, excluding runtime artifacts."""

    if not path.is_dir():
        raise ValueError(f"controller source directory does not exist: {path}")
    files = [
        item for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pth", ".pt", ".ckpt"}
    ]
    if not files:
        raise ValueError(f"controller source directory is empty: {path}")
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda entry: entry.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def parse_checkpoint_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(
                f"checkpoint {value!r} must have LABEL=/absolute/path format")
        if label in result:
            raise ValueError(f"duplicate checkpoint label {label!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"checkpoint does not exist: {path}")
        result[label] = path
    return result


def _reject_privileged_fields(payload: Mapping[str, Any]) -> None:
    leaked = sorted(
        str(key) for key in payload
        if str(key).strip().lower() in FORBIDDEN_RUNTIME_FIELDS)
    if leaked:
        raise ValueError(
            "privileged runtime fields are forbidden: " + ", ".join(leaked))


def _validate_pointgoal_batch(raw_goal_data: str) -> list[dict[str, list[float]]]:
    try:
        payload = json.loads(raw_goal_data)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("goal_data must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("goal_data must be a JSON object")
    _reject_privileged_fields(payload)
    goal_x = payload.get("goal_x")
    goal_y = payload.get("goal_y")
    if (not isinstance(goal_x, list) or not isinstance(goal_y, list)
            or len(goal_x) != len(goal_y) or not goal_x):
        raise ValueError("goal_x and goal_y must be equal non-empty lists")
    return [
        fixed_bearing_payload([forward, left])
        for forward, left in zip(goal_x, goal_y)
    ]


def _validate_trajectory_payload(payload: Mapping[str, Any]) -> None:
    for key in ("trajectory", "all_trajectory", "all_values"):
        if key not in payload:
            raise ValueError(f"upstream response is missing {key!r}")
        try:
            value = np.asarray(payload[key], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"upstream {key} is not numeric") from exc
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"upstream {key} must be non-empty and finite")
        if key in {"trajectory", "all_trajectory"}:
            if value.ndim < 2 or value.shape[-1] != 3:
                raise ValueError(
                    f"upstream {key} must end in trajectory xyz, got {value.shape}")


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from exc
    if value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_cec_takeover(
    *,
    adapter: str,
    files: Mapping[str, tuple[str, bytes, str]],
    form: Mapping[str, Any],
) -> str:
    if form.get("cec_action_authorized") != "1":
        raise ValueError("CEC accepted action must carry authorization")
    proof_sha = _validate_sha256(form.get("cec_proof_sha256"), "CEC proof")
    try:
        selected_anchor = int(form.get("cec_selected_anchor"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CEC takeover requires a selected anchor") from exc
    if selected_anchor < 0:
        raise ValueError("CEC selected anchor must be non-negative")
    if adapter == "verified_anchor_imagegoal":
        if form.get("goal_source") != "certified_history_anchor":
            raise ValueError(
                "verified_anchor_imagegoal takeover goal must be a "
                "certified history anchor")
        advertised = _validate_sha256(
            form.get("cec_anchor_sha256"), "CEC anchor")
        actual = hashlib.sha256(files["goal"][1]).hexdigest()
        if actual != advertised:
            raise ValueError(
                "goal bytes do not match the certified anchor")
    return proof_sha


@dataclass(frozen=True)
class ProxyConfig:
    comparison: ComparisonPlan
    repo_root: Path
    upstream_base: str
    checkpoints: Mapping[str, Path]
    timeout_s: float = 180.0


class ControllerPortabilityProxy:
    def __init__(self, config: ProxyConfig, *, session=None):
        self.config = config
        self.spec = validate_comparison_plan(config.comparison)
        if self.spec.key not in EXPECTED_ALGO:
            raise ValueError(
                f"no audited proxy mapping for controller {self.spec.key!r}")
        if not config.upstream_base.startswith(("http://", "https://")):
            raise ValueError("upstream_base must be an HTTP(S) URL")
        if not np.isfinite(config.timeout_s) or config.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")

        expected_labels = REQUIRED_CHECKPOINT_LABELS[self.spec.key]
        actual_labels = frozenset(config.checkpoints)
        if actual_labels != expected_labels:
            raise ValueError(
                f"{self.spec.display_name} checkpoint labels {sorted(actual_labels)} "
                f"do not match {sorted(expected_labels)}")
        checkpoint_hashes = {}
        for label, path in config.checkpoints.items():
            resolved = Path(path).expanduser().resolve()
            if not resolved.is_file():
                raise ValueError(f"checkpoint does not exist: {resolved}")
            checkpoint_hashes[label] = sha256_file(resolved)

        if self.spec.local_path is None:
            raise ValueError(f"{self.spec.display_name} has no local adapter")
        source_path = (config.repo_root / self.spec.local_path).resolve()
        self._identity = {
            "controller": self.spec.key,
            "controller_display_name": self.spec.display_name,
            "controller_family": self.spec.controller_family,
            "official_repository": self.spec.official_repository,
            "official_commit": self.spec.official_commit,
            "local_source_tree_sha256": sha256_source_tree(source_path),
            "checkpoint_sha256": checkpoint_hashes,
            "task_interfaces": sorted(self.spec.task_interfaces),
            "required_observations": sorted(self.spec.required_observations),
            "comparison_protocol": config.comparison.protocol,
            "depth_source": config.comparison.depth_source,
            "query_population": config.comparison.query_population,
            "reject_policy": config.comparison.reject_policy,
            "fallback_controller": config.comparison.fallback_controller,
            "cec_accept_adapter": self.spec.cec_accept_adapter,
            "role_label_visible": False,
            "uses_oracle_pose": False,
        }
        self._session = session if session is not None else requests.Session()
        self._reset_count = 0
        self._step_count = 0
        self._observation_count = 0

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "schema": PROXY_SCHEMA,
            "contract_schema_version": CONTROLLER_PORTABILITY_SCHEMA_VERSION,
            "upstream_base": self.config.upstream_base.rstrip("/"),
            **self._identity,
        }

    def _receipt(self, *, endpoint: str, upstream_algo: str) -> dict[str, Any]:
        return {
            "schema": PROXY_SCHEMA,
            "endpoint": endpoint,
            "upstream_algo": upstream_algo,
            "reset_count": self._reset_count,
            "step_count": self._step_count,
            "observation_count": self._observation_count,
            "pointgoal_frame": (
                "forward_left" if endpoint == "pointgoal_step" else None),
            "pointgoal_radius_m": (
                2.5 if endpoint == "pointgoal_step" else None),
            **self._identity,
        }

    def _post(self, endpoint: str, **kwargs):
        response = self._session.post(
            f"{self.config.upstream_base.rstrip('/')}/{endpoint}",
            timeout=float(self.config.timeout_s),
            **kwargs,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, MutableMapping):
            raise ValueError("upstream response must be a JSON object")
        return payload

    def reset(self, payload: Mapping[str, Any], *, env_only: bool = False):
        _reject_privileged_fields(payload)
        endpoint = "navigator_reset_env" if env_only else "navigator_reset"
        result = self._post(endpoint, json=dict(payload))
        expected_algo = EXPECTED_ALGO[self.spec.key]
        if result.get("algo") != expected_algo:
            raise ValueError(
                f"upstream algo {result.get('algo')!r}, expected {expected_algo!r}")
        self._reset_count += 1
        result["portability_receipt"] = self._receipt(
            endpoint=endpoint, upstream_algo=expected_algo)
        return result

    def step(
        self,
        endpoint: str,
        *,
        files: Mapping[str, tuple[str, bytes, str]],
        form: Mapping[str, Any],
    ) -> dict[str, Any]:
        if endpoint not in {"imagegoal_step", "pointgoal_step"}:
            raise ValueError(f"unsupported step endpoint {endpoint!r}")
        task = "imagegoal" if endpoint == "imagegoal_step" else "pointgoal"
        if task not in self.spec.task_interfaces:
            raise ValueError(
                f"{self.spec.display_name} does not support {task}")
        if endpoint == "imagegoal_step" and (
                self.config.comparison.protocol not in {
                    NATIVE_IMAGEGOAL, CEC_PROOF_HYBRID}):
            raise ValueError(
                "imagegoal_step is restricted to native or CEC-proof plans")
        if endpoint == "pointgoal_step" and (
                self.config.comparison.protocol not in {
                    CEC_BEARING_EXECUTOR, CEC_PROOF_HYBRID}):
            raise ValueError(
                "pointgoal_step is restricted to CEC bearing/proof plans")
        controller_native_fallback = bool(
            self.config.comparison.protocol == CEC_PROOF_HYBRID
            and self.config.comparison.reject_policy
            == "controller_native_exact"
            and endpoint == "imagegoal_step"
            and form.get("cec_action_authorized") in (None, "")
        )
        if (self.config.comparison.protocol == CEC_PROOF_HYBRID
                and not controller_native_fallback):
            expected_endpoint = (
                "imagegoal_step"
                if self.spec.cec_accept_adapter == "verified_anchor_imagegoal"
                else "pointgoal_step"
            )
            if endpoint != expected_endpoint:
                raise ValueError(
                    f"{self.spec.display_name} CEC adapter requires "
                    f"{expected_endpoint}")

        _reject_privileged_fields(form)
        required_files = {"image"}
        if "depth" in self.spec.required_observations:
            required_files.add("depth")
        if endpoint == "imagegoal_step":
            required_files.add("goal")
        missing = sorted(required_files - set(files))
        if missing:
            raise ValueError("missing request files: " + ", ".join(missing))
        if endpoint == "pointgoal_step":
            _validate_pointgoal_batch(form.get("goal_data"))
        proof_sha = None
        if (self.config.comparison.protocol == CEC_PROOF_HYBRID
                and not controller_native_fallback):
            proof_sha = _validate_cec_takeover(
                adapter=self.spec.cec_accept_adapter,
                files=files,
                form=form,
            )

        result = self._post(
            endpoint,
            files=dict(files),
            data=dict(form),
        )
        _validate_trajectory_payload(result)
        self._step_count += 1
        if proof_sha is not None:
            result["cec_proof_sha256"] = proof_sha
        result["portability_receipt"] = self._receipt(
            endpoint=endpoint,
            upstream_algo=EXPECTED_ALGO[self.spec.key],
        )
        return result

    def observe(
        self,
        *,
        files: Mapping[str, tuple[str, bytes, str]],
        form: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Advance only a short-context controller's causal RGB history
        during shared fallback (ViNT/GNM/NoMaD keep a memory_queue of the
        last few frames that must stay causal even while it is not deciding
        the action)."""
        if self.spec.key not in OBSERVATION_STEP_CONTROLLERS:
            raise ValueError(
                f"observation_step is not available for {self.spec.key!r}")
        if self.config.comparison.protocol != CEC_PROOF_HYBRID:
            raise ValueError("observation_step requires the CEC proof protocol")
        _reject_privileged_fields(form)
        if set(files) != {"image"}:
            raise ValueError("observation_step requires exactly one image")
        result = self._post(
            "observation_step", files=dict(files), data=dict(form))
        expected_algo = EXPECTED_ALGO[self.spec.key]
        if result.get("algo") != expected_algo or result.get("observed") is not True:
            raise ValueError("observation_step returned the wrong identity")
        self._observation_count += 1
        result["portability_receipt"] = self._receipt(
            endpoint="observation_step", upstream_algo=expected_algo)
        return result


def create_app(proxy: ControllerPortabilityProxy) -> Flask:
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return jsonify(proxy.health())

    @app.post("/navigator_reset")
    def navigator_reset():
        return jsonify(proxy.reset(request.get_json(force=True)))

    @app.post("/navigator_reset_env")
    def navigator_reset_env():
        return jsonify(proxy.reset(request.get_json(force=True), env_only=True))

    @app.post("/observation_step")
    def observation_step():
        files = {
            key: (value.filename or key, value.read(), value.mimetype)
            for key, value in request.files.items()
        }
        return jsonify(proxy.observe(
            files=files,
            form=request.form.to_dict(),
        ))

    def forward_step(endpoint: str):
        files = {
            key: (value.filename or key, value.read(), value.mimetype)
            for key, value in request.files.items()
        }
        return jsonify(proxy.step(
            endpoint,
            files=files,
            form=request.form.to_dict(),
        ))

    @app.post("/imagegoal_step")
    def imagegoal_step():
        return forward_step("imagegoal_step")

    @app.post("/pointgoal_step")
    def pointgoal_step():
        return forward_step("pointgoal_step")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--depth-source", required=True)
    parser.add_argument("--query-population", required=True)
    parser.add_argument("--reject-policy", required=True)
    parser.add_argument("--fallback-controller", default=None)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help="repeat LABEL=/absolute/checkpoint/path",
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    comparison = ComparisonPlan(
        controller=args.controller,
        protocol=args.protocol,
        depth_source=args.depth_source,
        query_population=args.query_population,
        reject_policy=args.reject_policy,
        fallback_controller=args.fallback_controller,
    )
    checkpoints = parse_checkpoint_arguments(args.checkpoint)
    proxy = ControllerPortabilityProxy(ProxyConfig(
        comparison=comparison,
        repo_root=args.repo_root.expanduser().resolve(),
        upstream_base=args.upstream_base,
        checkpoints=checkpoints,
        timeout_s=args.timeout_s,
    ))
    create_app(proxy).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
