import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robomme_example() -> dict:
    return {
        "observation/front_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/joint_state": np.random.rand(7),
        "observation/gripper_state": np.random.rand(1),
        "prompt": "counting task instruction",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_state = np.asarray(data["observation/gripper_state"])
        if gripper_state.ndim == 0:
            gripper_state = gripper_state[np.newaxis]
        
        joint_state = np.asarray(data["observation/joint_state"])
        state = np.concatenate([joint_state, gripper_state])

        front_image = _parse_image(data["observation/front_rgb"])
        wrist_image = _parse_image(data["observation/wrist_rgb"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (front_image, wrist_image, np.zeros_like(front_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (front_image, np.zeros_like(front_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMMEOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
