"""Compare two predicted position readouts after the same original certificate."""
import hashlib
import json
import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    from gem.sparse import SparseReadout
    root = Path(__file__).resolve().parents[1]
    run = Path(os.environ["LINGBOT_POSE_LOG"]).parent
    original = SparseReadout.read_sparse

    def read(self, goal_jpg_bytes, candidates, **kwargs):
        config = json.loads((run / "active_position_readout.json").read_text())
        if (config["goal_key"] != hashlib.md5(goal_jpg_bytes).hexdigest()
                or config["arm"] not in ("pnp_goal", "historical_camera")):
            raise RuntimeError("Unexpected goal or position-readout arm")
        if (kwargs.get("guidance_mode", "endpoint_bearing") != "endpoint_bearing"
                or kwargs.get("authority_policy", "strict_certificate") != "strict_certificate"
                or kwargs.get("graph_rescue", False) or kwargs.get("allow_learned_rescue", False)):
            raise RuntimeError("Position experiment changed the downstream method")
        result = dict(original(self, goal_jpg_bytes, candidates, **kwargs))
        pnp_bearing = result.get("direction_vector")
        if config["arm"] == "historical_camera" and result.get("accepted") is True:
            anchor = result["selected_anchor"]
            pose = self.poses[anchor].detach().float().cpu().numpy()
            direction = self.backend._certified_bearing_vector(pose)
            result["direction_vector"] = result["aux_pose"] = direction
        row = dict(arm=config["arm"], goal_key=config["goal_key"],
                   frame_idx=result.get("frame_idx"), accepted=result.get("accepted"),
                   selected_anchor=result.get("selected_anchor"), candidates=candidates,
                   pnp_bearing=pnp_bearing, consumed_bearing=result.get("direction_vector"),
                   cached=result.get("cached"), certificate=result.get("certificate"),
                   pnp=result.get("pnp"))
        with (run / "position_readout_audit.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        return result

    SparseReadout.read_sparse = read
    runpy.run_path(str(root / "MemNavData/lingbot_pose_diagnostic_server.py"), run_name="__main__")
