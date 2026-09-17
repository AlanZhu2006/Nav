"""Released goal-conditioned ViNT/NoMaD inference in the shared action domain.

RGB in, raw goal-conditioned waypoints out. NoMaD uses mask=0; mask=1 is
exploration. The NavDP baseline wrapper's distance>7 trajectory suppression
is not part of the released navigation policy and is not applied here.
No model is trained or substituted on failure.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


def load_agent(controller, root, checkpoint, device):
    if controller not in ("vint", "nomad"):
        raise ValueError(controller)
    package = Path(root) / "NavDP/baselines" / controller
    # The two baseline folders both define base_agent; one model per process.
    sys.path.insert(0, str(package))
    common = dict(image_intrinsic=np.eye(3), model_path=str(checkpoint),
        model_config_path=str(package / "configs" / f"{controller}.yaml"),
        robot_config_path=str(package / "configs/robot_config.yaml"), device=device)
    if controller == "vint":
        from vint_agent import ViNTAgent, transform_images
        agent = ViNTAgent(**common)
    else:
        from nomad_agent import NoMaDAgent, transform_images
        agent = NoMaDAgent(**common, data_config_path=str(package / "configs/data_config.yaml"))
    return agent, transform_images


def predict(agent, transform, controller, image, goal, *, seed, sample_num=8):
    """Advance the selected policy's FIFO once and consume paired noise once."""
    if image.shape[0] != 1 or goal.shape[0] != 1 or agent.batch_size != 1:
        raise ValueError("The paired evaluator is single-environment")
    with torch.no_grad():
        agent.callback_obs(image)
        observations = torch.cat([transform(frames, agent.image_size, center_crop=False).to(agent.device)
                                  for frames in agent.memory_queue], dim=0)
        target = torch.cat([transform(Image.fromarray(frame), agent.image_size, center_crop=False).to(agent.device)
                            for frame in goal], dim=0)
        if controller == "vint":
            distances, waypoints = agent.vint_former.predict_imagegoal_distance_and_action(observations, target)
            if agent.normalize:
                waypoints[:, :, :2] *= agent.MAX_V / agent.RATE
            trajectories = agent.traj_generate.TrajGeneratorFromPFreeRot(waypoints[:, :, :3], step=.1)[:, None]
            mask = None
        elif controller == "nomad":
            torch.manual_seed(int(seed))
            if str(agent.device).startswith("cuda"):
                torch.cuda.manual_seed_all(int(seed))
            mask = torch.zeros(1, dtype=torch.long, device=agent.device)
            distances, condition = agent.nomad_former.predict_imagegoal_distance(observations, target, mask)
            actions = agent.nomad_former.predict_imagegoal_action(condition, sample_num=sample_num)
            waypoints = agent.get_action(actions, agent.ACTION_STATS)
            if agent.normalize:
                waypoints[:, :, :2] *= agent.MAX_V / agent.RATE
            trajectories = agent.traj_generate.TrajGeneratorFromPFreeRot(waypoints[:, :, :2], step=.1)
            trajectories = trajectories.reshape(1, sample_num, trajectories.shape[1], 2)
            trajectories = torch.cat((trajectories, torch.zeros_like(trajectories[..., :1])), dim=-1)
        else:
            raise ValueError(controller)
        if not torch.isfinite(trajectories).all() or not torch.isfinite(distances).all():
            raise RuntimeError("Nonfinite frozen-controller output")
        return dict(trajectory=trajectories[:, 0].cpu().numpy().tolist(),
                    all_trajectory=trajectories.cpu().numpy().tolist(),
                    all_values=np.zeros((1, trajectories.shape[1])).tolist(),
                    predicted_temporal_distance=distances.cpu().numpy().reshape(-1).tolist(),
                    distance_used_to_suppress_trajectory=False, selected_sample_index=0,
                    goal_mask=None if mask is None else mask.cpu().tolist(),
                    diffusion_seed=int(seed), controller_seed_consumed=controller == "nomad",
                    policy_pointgoal_consumed=False, controller_depth_source="none")
