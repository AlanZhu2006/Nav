import inspect

import pytest

from MemNavData.image_controller_goal_adapter import reset_source, replay_source


@pytest.mark.parametrize("controller", ["vint", "nomad"])
def test_reset_bridge_only_changes_downstream_validation(controller):
    from pathlib import Path
    import ast
    path = Path(__file__).with_name("eval_2leg_habitat.py")
    source = path.read_text()
    parsed = ast.parse(source)
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "srv_reset")
    original = ast.get_source_segment(source, function)
    updated = reset_source(original, controller)
    compile(updated, "<test>", "exec")
    assert updated.count(f'novel_info.get("controller") != {controller!r}') == 1
    assert 'novel_info.get("controller_depth_source") != "none"' in updated
    assert updated.count('"depth_source": args.navdp_depth_source') == 1
    assert 'memnav_info.get("monocular_depth")' in updated


def test_unknown_entry_fails():
    with pytest.raises(ValueError):
        reset_source("def missing(): pass", "nomad")


def test_source_keeps_goal_and_noise_contract():
    from pathlib import Path
    source = Path(__file__).with_name("image_controller_policy.py").read_text()
    assert 'mask = torch.zeros(1' in source
    assert 'torch.manual_seed(int(seed))' in source
    assert 'distance_used_to_suppress_trajectory=False' in source
    assert 'trajectories[:, 0]' in source


def test_replay_preserves_history_and_models_padded_context():
    import ast
    from pathlib import Path
    source = Path(__file__).with_name("shared_online_double_revisit_runtime.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "replay_online_a")
    original = ast.get_source_segment(source, node)
    updated = replay_source(original)
    compile(updated, "<replay-test>", "exec")
    assert updated.replace("expected_queue = final_memory_size  # repeated-first-frame context",
        "expected_queue = min(len(plan_steps), final_memory_size)") == original
    assert 'sha256_file(image) == pose["jpg_sha256"]' in updated
