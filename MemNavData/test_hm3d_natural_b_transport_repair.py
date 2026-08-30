from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "MemNavData/freeze_hm3d_natural_b_transport_repair.py"
VERIFIER = ROOT / "MemNavData/independent_verify_hm3d_natural_b_transport_repair.py"
SHARD = ROOT / "MemNavData/slurm_hm3d_natural_b_transport_repair_shard.sbatch"
LAUNCHER = ROOT / "MemNavData/slurm_hm3d_natural_b_transport_repair_launch.sbatch"


def seal(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def test_missing_membership_is_identity_only_and_partials_are_preserved(tmp_path: Path):
    episodes = [
        {"scene": f"scene{i // 3}", "episode": f"episode_{i:04d}",
         "final14_scene_rank": i // 3}
        for i in range(84)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": episodes}))
    manifest_sha = seal(manifest)
    shards = [{
        "shard_index": i, "history_indices": [i], "history_count": 1,
        "navigation_outcomes_read": False,
    } for i in range(84)]
    schedule = tmp_path / "shards.json"
    schedule.write_text(json.dumps({
        "candidate_histories": 84, "all_candidates_partitioned_once": True,
        "navigation_outcomes_read": False, "query_policy_outcomes_read": False,
        "shards": shards,
    }))
    schedule_sha = seal(schedule)
    run = tmp_path / "run"
    for index, row in enumerate(episodes):
        if index in (1, 4):
            continue
        out = run / "factual_b" / f"{index:03d}_{row['scene']}_{row['episode']}"
        out.mkdir(parents=True)
        completion = out / "completion.json"
        completion.write_text(json.dumps({"reached_B": index % 2 == 0}))
        seal(completion)
    partial = run / "factual_b/001_scene0_episode_0001"
    partial.mkdir(parents=True)
    (partial / "opaque.log").write_text("do not interpret me")
    plan = run / "transport_repair/plan.json"
    archive = run / "failed_attempts/transport_repair"
    subprocess.run([
        sys.executable, str(FREEZER), "--run-root", str(run),
        "--benchmark-manifest", str(manifest),
        "--expected-benchmark-sha256", manifest_sha,
        "--shard-manifest", str(schedule),
        "--expected-shard-sha256", schedule_sha,
        "--archive-root", str(archive), "--out", str(plan),
    ], check=True)
    payload = json.loads(plan.read_text())
    assert payload["missing_history_indices"] == [1, 4]
    assert payload["partial_history_indices"] == [1]
    assert payload["navigation_outcomes_read"] is False
    assert not partial.exists()
    assert (archive / partial.name / "opaque.log").read_text() == "do not interpret me"
    verification = run / "transport_repair/verification.json"
    subprocess.run([
        sys.executable, str(VERIFIER), "--run-root", str(run),
        "--benchmark-manifest", str(manifest), "--repair-plan", str(plan),
        "--archive-root", str(archive), "--out", str(verification),
    ], check=True)
    assert json.loads(verification.read_text())["verified"] is True


def test_repair_runtime_preserves_the_frozen_scientific_contract():
    shard = SHARD.read_text()
    launcher = LAUNCHER.read_text()
    assert 'MAX_STEPS=600' in shard
    assert 'MODE=lifelong_b' in shard
    assert 'FORMAL_INDICES_OVERRIDE="${row[1]}"' in shard
    assert 'RUNTIME_ATTEMPT="${REPAIR_RUNTIME_PREFIX}' in shard
    assert 'bash "${WRAPPER_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"' in shard
    assert '--dependency="afterok:${r_job}"' in launcher
    assert '--partition="${GPU_PARTITION}"' in launcher
    assert 'repair_root=${RUN_ROOT}/${REPAIR_TAG}' in launcher
    assert 'fallback_completion_allowed\':False' in launcher
    assert 'navigation_outcomes_read\':False' in launcher
