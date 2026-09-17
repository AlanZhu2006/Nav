"""Real checkpoint HTTP/reset/goal/paired-noise audit without Habitat or GPU."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.image_controller_policy import load_agent
from MemNavData.image_controller_repaired_server import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("vint", "nomad"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    started = time.perf_counter()
    agent, transform = load_agent(args.controller, ROOT, args.checkpoint, "cpu")
    app = create_app(args.controller, agent, transform)
    app.testing = True
    client = app.test_client()
    frames = [(args.history / f"{i:06d}.jpg").read_bytes() for i in range(7)]
    goal = args.goal.read_bytes()

    def execute(seed):
        reset = client.post("/navigator_reset", json=dict(batch_size=1, intrinsic=np.eye(3).tolist())).get_json()
        assert reset["controller_depth_source"] == "none"
        for i, rgb in enumerate(frames[:-1]):
            r = client.post("/memory_replay_step", data={"image": (io.BytesIO(rgb), "rgb.jpg")}).get_json()
            assert r["diffusion_sampled"] is False
            assert r["execution_input_audit"]["observations"] == i + 1
            assert r["memory_size"] == agent.memory_size + 1
            assert r["queue_lengths"] == [agent.memory_size + 1]
            assert r["queue_padding_strategy"] == "repeat_first_observation"
        result = client.post("/imagegoal_step", data=dict(image=(io.BytesIO(frames[-1]), "rgb.jpg"),
            goal=(io.BytesIO(goal), "goal.jpg"), diffusion_seed=str(seed))).get_json()
        audit = result["execution_input_audit"]
        assert audit["observations"] == 7 and audit["received_file_fields"] == ["goal", "image"]
        for entry, encoded in zip(audit["image_calls"], (frames[-1], goal)):
            rgb = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))[None]
            assert entry["input_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
        return result

    a, b = execute(7), execute(7)
    np.testing.assert_array_equal(a["all_trajectory"], b["all_trajectory"])
    goal_effect = None
    if args.controller == "nomad":
        assert a["goal_mask"] == [0] and a["controller_seed_consumed"] is True
        c = execute(8)
        assert not np.array_equal(a["all_trajectory"], c["all_trajectory"])
        observation = torch.cat([transform(q, agent.image_size, center_crop=False) for q in agent.memory_queue])
        goals = [torch.cat([transform(Image.open(io.BytesIO(g)).convert("RGB"), agent.image_size, center_crop=False)])
                 for g in (goal, frames[0])]
        differences = {}
        for mask in (0, 1):
            vectors = [agent.nomad_former.predict_imagegoal_distance(observation, g, torch.tensor([mask]))[1]
                       for g in goals]
            differences[str(mask)] = float((vectors[0] - vectors[1]).abs().max())
        assert differences["0"] > 1e-6 and differences["1"] < 1e-5
        goal_effect = differences
    report = dict(verified=True, controller=args.controller, device="cpu",
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        http_rgb_exact=True, reset_replay_equal=True, equal_seed_all_candidates_equal=True,
        controller_seed_consumed=a["controller_seed_consumed"], goal_mask=a["goal_mask"],
        goal_change_condition_max_abs_by_mask=goal_effect,
        distance_used_to_suppress_trajectory=a["distance_used_to_suppress_trajectory"],
        predicted_temporal_distance=a["predicted_temporal_distance"],
        actual_trajectory_nonzero=bool(np.max(np.abs(a["trajectory"])) > 0),
        wall_seconds=time.perf_counter()-started, scope="CPU contract, not navigation SR")
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
