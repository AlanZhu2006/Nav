"""Read-only import provenance and CLI checks in the actual runtime interpreters."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def check(kind):
    modules = ["MemNavData.bounded_pursuit", "MemNavData.navdp_front_goal_adapter",
               "MemNavData.lingbot_depth_raster", "MemNavData.monocular_depth_runtime",
               "MemNavData.run_repaired_fullmono_local"]
    if kind == "habitat":
        for name in ("habitat_sim", "magnum", "quaternion", "cv2", "pandas", "pyarrow", "requests"):
            importlib.import_module(name)
        modules += ["MemNavData.verify_repaired_fullmono_local", "MemNavData.materialize_hm3d_fullmono_online_a",
                    "MemNavData.build_final14_role_pair_scene", "generate_twoleg"]
    else:
        modules += ["policy_agent", "policy_network", "policy_backbone"]
        if kind == "memnav":
            modules += ["internnav.model.basemodel.memnav.memnav_policy", "scripts.train.configs.memnav"]
        else:
            modules += ["depth_anything.depth_anything_v2.dpt"]
    resolved = {}
    for name in modules:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"{name} escaped source bundle: {path}")
        if name.startswith("policy_") and not path.is_relative_to(ROOT / "NavDP/baselines" / kind):
            raise ValueError(f"{name} resolved to another server's sibling: {path}")
        resolved[name] = str(path.relative_to(ROOT))
    commands = 0
    if kind == "habitat":
        from MemNavData.run_repaired_fullmono_local import evaluator_command
        source = {"scene": "preflight", "episode": "episode_0000", "asset": "/preflight/scene.glb",
                  "source_episode": "/preflight/episodes/preflight/episode_0000", "seed": 2026082200}
        settings = [("native", None)] + [(arm, role) for role in ("novel", "revisit")
                                                   for arm in ("native", "raw_fixed", "cec")]
        for arm, role in settings:
            command = evaluator_command(source, Path("/preflight/no_output"), 21010, 21011,
                                        arm=arm, role=role, benchmark=Path("/preflight/benchmark"))
            command.append("--contract_dry_run")
            subprocess.run(command, check=True, env=os.environ.copy(), cwd=ROOT)
            commands += 1
    print(json.dumps({"verified": True, "kind": kind, "python": sys.executable,
                      "modules": resolved, "cli_combinations": commands}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("habitat", "memnav", "navdp"))
    check(parser.parse_args().kind)
