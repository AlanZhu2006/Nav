import hashlib
import pickle
import sys
import types
import numpy as np
import torch
from base_agent import GNMBaseAgent
from gnm_model import GNMPolicy, NoGoalGNMPolicy
from PIL import Image as PILImage
from typing import List
from torchvision import transforms
import traj_opt
import torchvision.transforms.functional as TF

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (4 / 3)

# The upstream GNM release is a training bundle ({"model": <GNM nn.Module>,
# "optimizer": ..., "scheduler": ...}) pickled by a training codebase this
# repo does not vendor.  Keep the unsafe compatibility path hash-pinned, same
# convention as vint_agent.py's _load_vint_state_dict; converted/plain
# state_dict checkpoints stay on weights_only.
_TRUSTED_LEGACY_CHECKPOINTS = {
    "4b03e0255f8a547290d4079f4e7d610ff69987122f17e019bd36684c08b3ee95",
}


class _DiscardedModel(torch.nn.Module):
    """Placeholder used only to unpickle a trusted bundle's model object.

    Pickle reconstructs instances via __new__ and a raw __dict__ update, so
    an nn.Module subclass with no real layers still ends up with the correct
    populated _parameters/_buffers/_modules and yields a genuine state_dict()
    -- the class identity is irrelevant to torch's unpickler."""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_module_aliases():
    modules = {
        name: types.ModuleType(name)
        for name in ("vint_train", "vint_train.models",
                     "vint_train.models.gnm", "vint_train.models.gnm.gnm")
    }
    modules["vint_train.models.gnm.gnm"].GNM = _DiscardedModel
    return modules


def _load_gnm_state_dict(model_path: str):
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint_sha = _sha256(model_path)
        if checkpoint_sha not in _TRUSTED_LEGACY_CHECKPOINTS:
            raise RuntimeError(
                "Refusing to unpickle an untrusted legacy GNM checkpoint "
                f"with sha256={checkpoint_sha}"
            )
        aliases = _legacy_module_aliases()
        previous = {name: sys.modules.get(name) for name in aliases}
        sys.modules.update(aliases)
        try:
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if hasattr(payload, "module"):
        payload = payload.module
    if hasattr(payload, "state_dict"):
        payload = payload.state_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported GNM checkpoint payload: {type(payload)!r}")
    return payload

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data
def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

class GNMAgent(GNMBaseAgent):
    def __init__(
        self,
        image_intrinsic,
        model_path: str,
        model_config_path: str,
        robot_config_path: str,
        device="cuda:0",
    ):
        super(GNMAgent, self).__init__(image_intrinsic, model_path, model_config_path, robot_config_path, device)
        self.gnm_former = GNMPolicy(self.cfg)
        self.gnm_former.to(self.device)
        self.gnm_former.model.load_state_dict(
            _load_gnm_state_dict(self.model_path), strict=True)
        self.gnm_former.eval()
        self.traj_generate = traj_opt.TrajOpt()

    def observe(self, image):
        """Advance only the causal RGB context without sampling an action."""
        self.callback_obs(image)

    def step_imagegoal(self, goal_image, image):
        with torch.no_grad():
            self.callback_obs(image)
            # [N, 3, h, w]
            goal_image = [
                transform_images(PILImage.fromarray(g_img), self.image_size, center_crop=False).to(self.device) for g_img in goal_image
            ]
            goal_image = torch.concat(goal_image, dim=0)
            # [N, N*3, h, w]
            input_image = [
                transform_images(imgs,self.image_size, center_crop=False).to(self.device) for imgs in self.memory_queue
            ]
            input_image = torch.concat(input_image, dim=0)
            distances, waypoints = self.gnm_former.predict_imagegoal_distance_and_action(input_image, goal_image)
            if self.normalize:
                waypoints[:,:,:2] *= self.MAX_V / self.RATE
            stop_mask = (distances > 7.0).unsqueeze(1).float()
            trajectory = self.traj_generate.TrajGeneratorFromPFreeRot(waypoints[:,:,0:3],step=0.1) * stop_mask
            return waypoints[:,:,0:3],trajectory
        
    def step_nogoal(self, image):
        with torch.no_grad():
            self.callback_obs(image)
            fake_goal = torch.randn((image.shape[0], 3, self.image_size[1], self.image_size[0])).to(self.device)
            # [N, N*3, h, w]
            input_image = [
                transform_images(imgs,self.image_size, center_crop=False).to(self.device) for imgs in self.memory_queue
            ]
            input_image = torch.concat(input_image, dim=0)
            distances, waypoints = self.gnm_former.predict_imagegoal_distance_and_action(input_image, fake_goal)
            if self.normalize:
                waypoints[:,:,:2] *= self.MAX_V / self.RATE
            trajectory = self.traj_generate.TrajGeneratorFromPFreeRot(waypoints[:,:,0:3],step=0.1)
            return waypoints[:,:,0:3],trajectory
        
class NoGoalGNMAgent(GNMBaseAgent):
    def __init__(
        self,
        image_intrinsic,
        model_path: str,
        model_config_path: str,
        robot_config_path: str,
        device="cuda:0",
    ):
        super(NoGoalGNMAgent, self).__init__(image_intrinsic, model_path, model_config_path, robot_config_path, device)
        self.gnm_former = NoGoalGNMPolicy(self.cfg)
        self.gnm_former.to(self.device)
        self.gnm_former.model.load_state_dict(
            _load_gnm_state_dict(self.model_path), strict=True)
        self.gnm_former.eval()
        self.traj_generate = traj_opt.TrajOpt()

    def step_nogoal(self, image):
        with torch.no_grad():
            self.callback_obs(image)
            # [N, N*3, h, w]
            input_image = [
                transform_images(imgs,self.image_size, center_crop=False).to(self.device) for imgs in self.memory_queue
            ]
            input_image = torch.concat(input_image, dim=0)
            distances, waypoints = self.gnm_former.predict_nogoal_distance_and_action(input_image)
            if self.normalize:
                waypoints[:,:,:2] *= self.MAX_V / self.RATE
            trajectory = self.traj_generate.TrajGeneratorFromPFreeRot(waypoints[:,:,0:3],step=0.1)
            return waypoints[:,:,0:3],trajectory
    

