#!/usr/bin/env python3
"""Observe LingBot's existing per-frame pose; do not correct or re-infer it."""
import hashlib
import json
import os
from pathlib import Path
import runpy


def pose_receipt(agent, image, index):
    # The first S-1 append calls intentionally buffer RGB without a pose.
    # Report this normal warm-up state instead of indexing an empty cache.
    pose = agent.cam_pose[-1] if agent.cam_pose else None
    return {"frame_idx": int(index), "image_sha256": hashlib.sha256(image).hexdigest(),
            "camera_pose9": None if pose is None else pose.detach().float().cpu().numpy().tolist(),
            "pose_count": len(agent.cam_pose), "monocular_depth": agent.monocular_depth_status(),
            "motion_receipt_recorded": agent.executor_motion_receipts[-1]}


def main():
    from MemNavData.multipart_crlf_repair import install
    install()
    from policy_agent import MemNavAgent

    output = Path(os.environ["LINGBOT_POSE_LOG"])
    original = MemNavAgent.add_frame

    def append(self, image, **kwargs):
        index = original(self, image, **kwargs)
        # The synchronous append is already complete. Reading this tiny tensor
        # does not alter the cache, gate, estimator, or returned receipt.
        row = pose_receipt(self, image, index)
        with output.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return index

    MemNavAgent.add_frame = append
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "NavDP/baselines/memnav/memnav_server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
