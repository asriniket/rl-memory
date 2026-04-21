import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robomme_example(num_memory_frames: int = 1) -> dict:
    if num_memory_frames < 1:
        raise ValueError(f"num_memory_frames must be >= 1, got {num_memory_frames}")
    if num_memory_frames == 1:
        front_shape = (256, 256, 3)
        wrist_shape = (256, 256, 3)
        joint_shape = (7,)
        gripper_shape = (1,)
    else:
        front_shape = (num_memory_frames, 256, 256, 3)
        wrist_shape = (num_memory_frames, 256, 256, 3)
        joint_shape = (num_memory_frames, 7)
        gripper_shape = (num_memory_frames, 1)
    return {
        "observation/front_rgb": np.random.randint(256, size=front_shape, dtype=np.uint8),
        "observation/wrist_rgb": np.random.randint(256, size=wrist_shape, dtype=np.uint8),
        "observation/joint_state": np.random.rand(*joint_shape),
        "observation/gripper_state": np.random.rand(*gripper_shape),
        "prompt": "counting task instruction",
    }


def _parse_image(image) -> np.ndarray:
    """Normalize a single image or a stack of K images into `(..., H, W, 3)` uint8 form."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # Reorient channel-first inputs. Work from the trailing axes so that the optional
    # leading memory axis is preserved.
    if image.ndim >= 3 and image.shape[-3] == 3 and image.shape[-1] != 3:
        if image.ndim == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        else:
            image = einops.rearrange(image, "... c h w -> ... h w c")
    return image


def _zero_like_camera(front_image: np.ndarray) -> np.ndarray:
    return np.zeros_like(front_image)


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_state = np.asarray(data["observation/gripper_state"])
        joint_state = np.asarray(data["observation/joint_state"])

        front_image = _parse_image(data["observation/front_rgb"])
        has_memory_axis = front_image.ndim == 4

        if has_memory_axis:
            if gripper_state.ndim == 1:
                gripper_state = gripper_state[:, np.newaxis]
            if joint_state.ndim == 1:
                joint_state = joint_state[np.newaxis]
            state = np.concatenate([joint_state, gripper_state], axis=-1)
        else:
            if gripper_state.ndim == 0:
                gripper_state = gripper_state[np.newaxis]
            state = np.concatenate([joint_state, gripper_state])

        wrist_image = _parse_image(data["observation/wrist_rgb"])
        zero_image = _zero_like_camera(front_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (front_image, wrist_image, zero_image)
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (front_image, zero_image, wrist_image)
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
