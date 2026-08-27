import math
import hashlib
import pickle
import torch
import yaml
import numpy as np
import torchvision.transforms as transforms
import traj_opt
from planner_net import PlannerNet


# The official iPlanner release serializes ``(PlannerNet, best_loss)`` rather
# than a plain state_dict.  Loading an arbitrary pickled module is unsafe, so
# the legacy path is enabled only for the audited upstream checkpoint.
_TRUSTED_LEGACY_CHECKPOINTS = {
    "685f16cde28d05249d50d24ed79ab4bdc94b3fbbcb99c8dbaed31039d11633b9",
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_dict(model_path: str):
    """Load a plain state_dict, or the hash-pinned official legacy release."""
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint_sha = _sha256(model_path)
        if checkpoint_sha not in _TRUSTED_LEGACY_CHECKPOINTS:
            raise RuntimeError(
                "Refusing to unpickle an untrusted legacy iPlanner checkpoint "
                f"with sha256={checkpoint_sha}"
            )
        payload = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(payload, (tuple, list)):
        payload = payload[0]
    if hasattr(payload, "state_dict"):
        payload = payload.state_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported iPlanner checkpoint payload: {type(payload)!r}")
    return payload

class IPlannerAgent():
    def __init__(
        self,
        image_intrinsic: torch.Tensor,
        model_path: str,
        model_config_path: str,
        device="cuda:0",
    ):
        self.image_intrinsic = image_intrinsic
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.device = device
        self.traj_generate = traj_opt.TrajOpt()
        self.load_model(self.model_path, self.model_config_path)
        self.transform = transforms.Resize(self.img_input_size, antialias=None)  # type: ignore

    def load_model(self, model_path: str, model_config_path: str):
        with open(model_config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.img_input_size = self.cfg["img_input_size"]
        self.sensor_offset_x = self.cfg["sensor_offset_x"]
        self.sensor_offset_y = self.cfg["sensor_offset_y"]
        self.max_depth = self.cfg["max_depth"]
        self.max_goal_distance = self.cfg["max_goal_distance"]
        self.is_traj_shift = False
        if math.hypot(self.sensor_offset_x, self.sensor_offset_y) > 1e-1:
            self.is_traj_shift = True
        self.net = PlannerNet(encoder_channel=16)
        model_state_dict = _load_state_dict(model_path)
        self.net.load_state_dict(model_state_dict, strict=True)
        self.net.eval()
        self.net = self.net.to(self.device)

    def process_depth(self, depth: torch.Tensor) -> torch.Tensor:
        # ``expand`` returns a shared-stride view; clone before the in-place
        # sanitization below so future PyTorch releases do not reject it.
        depth = self.transform(depth).expand(1, 3, -1, -1).clone()
        depth[depth > self.max_depth] = 0.0
        depth[~torch.isfinite(depth)] = 0  # set all inf or nan values to 0
        return depth
    
    def process_image(self, image: torch.Tensor) -> np.ndarray:
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
        return image_np
    
    def plan(self, image, goal_robot_frame):
        with torch.no_grad():
            keypoints, fear = self.net(self.process_depth(image), goal_robot_frame)
        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat(
                (torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), dim=1
            )
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear

    def step_pointgoal(
        self,
        dep_image: torch.Tensor,
        goal_robot_frame: torch.Tensor,
    ):
        with torch.no_grad():
            tensor_dep_image = torch.as_tensor(dep_image[:,:,:,0], device=self.device, dtype=torch.float32)
            tensor_goal_robot_frame = torch.as_tensor(goal_robot_frame[:,0:3], device=self.device, dtype=torch.float32)
            keypoints, traj, fear = self.plan(tensor_dep_image, tensor_goal_robot_frame)
            return keypoints, traj, fear
