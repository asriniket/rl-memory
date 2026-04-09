"""
Convert RoboMME HDF5 datasets to LeRobot format.

Usage:
uv run examples/robomme/convert_robomme_to_lerobot.py --h5_data_dir /path/to/robomme_data_h5
"""

import gc
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

REPO_NAME = "robomme_counting"

COUNTING_TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def decode_h5_string(raw) -> str:
    if isinstance(raw, np.ndarray):
        raw = raw.flatten()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    return raw


def main(
    h5_data_dir: str,
    *,
    tasks: list[str] = COUNTING_TASKS,
):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    h5_data_dir = Path(h5_data_dir)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=30,
        features={
            "front_rgb": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_rgb": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_state"],
            },
            "gripper_state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=0,
    )

    for task in tasks:
        h5_file = h5_data_dir / f"record_dataset_{task}.h5"
        if not h5_file.exists():
            print(f"Warning: {h5_file} not found, skipping")
            continue

        print(f"Processing task: {task}")
        with h5py.File(h5_file, "r") as f:
            episode_keys = sorted(
                [k for k in f.keys() if k.startswith("episode_")],
                key=lambda k: int(k.split("_")[1]),
            )
            
            for ep_key in tqdm(episode_keys, desc=f"Converting {task}"):
                episode = f[ep_key]
                setup = episode["setup"]
                
                task_goal_raw = setup["task_goal"][()]
                if isinstance(task_goal_raw, np.ndarray):
                    task_goal = decode_h5_string(task_goal_raw[0])
                else:
                    task_goal = decode_h5_string(task_goal_raw)

                timestep_keys = sorted(
                    [k for k in episode.keys() if k.startswith("timestep_")],
                    key=lambda k: int(k.split("_")[1]),
                )
                
                for ts_key in timestep_keys:
                    timestep = episode[ts_key]

                    if bool(timestep["info"]["is_video_demo"][()]):
                        continue

                    obs = timestep["obs"]
                    action = timestep["action"]
                    
                    front_rgb = np.asarray(obs["front_rgb"][()])
                    wrist_rgb = np.asarray(obs["wrist_rgb"][()])
                    joint_state = np.asarray(obs["joint_state"][()], dtype=np.float32)
                    gripper_state = np.asarray(obs["gripper_state"][()], dtype=np.float32)
                    
                    if gripper_state.shape[0] == 2:
                        gripper_state = gripper_state[:1]
                    
                    joint_action = np.asarray(action["joint_action"][()], dtype=np.float32)
                    
                    dataset.add_frame({
                        "front_rgb": front_rgb,
                        "wrist_rgb": wrist_rgb,
                        "joint_state": joint_state,
                        "gripper_state": gripper_state,
                        "actions": joint_action,
                        "task": task_goal,
                    })
                
                dataset.save_episode()
                dataset.hf_dataset = dataset.create_hf_dataset()
                gc.collect()

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
