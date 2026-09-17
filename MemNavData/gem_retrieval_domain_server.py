"""Private server entrypoint for the frozen archive-access experiment."""
import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    from gem import sparse
    from MemNavData.gem_retrieval_domain import install
    root = Path(__file__).resolve().parents[1]
    pose_log = Path(os.environ["LINGBOT_POSE_LOG"])
    install(sparse, pose_log.parent / "active_domain.json",
            pose_log.parent / "retrieval_domain_audit.jsonl")
    runpy.run_path(str(root / "MemNavData/lingbot_pose_diagnostic_server.py"),
                   run_name="__main__")
