"""RoboMME counting eval over websocket. Server: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_robomme_counting_lora --policy.dir=<checkpoint_dir>`."""

import collections
import dataclasses
import logging
import pathlib

import imageio
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder
import tqdm
import tyro

COUNTING_TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000

    dataset: str = "test"
    max_steps: int = 1300
    replan_steps: int = 8

    save_videos: bool = True
    video_dir: pathlib.Path = pathlib.Path("runs/eval_videos")
    video_fps: int = 30


def env_obs_to_policy_obs(obs: dict, task_goal: str) -> dict:
    front_rgb = np.asarray(obs["front_rgb_list"])
    wrist_rgb = np.asarray(obs["wrist_rgb_list"])
    joint_state = np.asarray(obs["joint_state_list"], dtype=np.float32)
    gripper_state = np.asarray(obs["gripper_state_list"], dtype=np.float32)
    if gripper_state.shape[-1] == 2:
        gripper_state = gripper_state[..., :1]

    return {
        "observation/front_rgb": front_rgb,
        "observation/wrist_rgb": wrist_rgb,
        "observation/joint_state": joint_state,
        "observation/gripper_state": gripper_state,
        "prompt": task_goal,
    }


def _grab_frame(obs: dict) -> np.ndarray:
    front = np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8)
    wrist = np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8)
    return np.hstack([front, wrist])


def main(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    results: dict[str, list[bool]] = {}

    for task in COUNTING_TASKS:
        logging.info("Evaluating task: %s", task)
        env_builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=args.dataset,
            action_space="joint_angle",
            max_steps=args.max_steps,
        )
        episode_count = env_builder.get_episode_num()
        task_successes: list[bool] = []

        for episode in tqdm.tqdm(range(episode_count), desc=task):
            env = env_builder.make_env_for_episode(episode)
            obs, info = env.reset()
            task_goal = info["task_goal"][0]

            frames: list[np.ndarray] = []
            if args.save_videos:
                frames.append(_grab_frame(obs))

            action_plan: collections.deque = collections.deque()
            outcome = "unknown"
            done = False

            while not done:
                if not action_plan:
                    policy_obs = env_obs_to_policy_obs(obs, task_goal)
                    action_chunk = client.infer(policy_obs)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, _reward, terminated, truncated, info = env.step(action)

                if args.save_videos:
                    frames.append(_grab_frame(obs))

                if info is not None and info.get("status") == "error":
                    logging.warning("[%s] Episode %s: env error", task, episode)
                    outcome = "error"
                    done = True

                elif terminated or truncated:
                    outcome = info.get("status", "unknown")
                    done = True

            task_successes.append(outcome == "success")

            if frames:
                video_path = args.video_dir / f"{task}_ep{episode}_{outcome}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(video_path), frames, fps=args.video_fps)

            env.close()

        results[task] = task_successes
        task_rate = sum(task_successes) / len(task_successes) if task_successes else 0.0
        logging.info("[%s] Success rate: %s/%s = %.3f", task, sum(task_successes), len(task_successes), task_rate)

    print("\n===== Results =====")
    all_successes: list[bool] = []
    for task in COUNTING_TASKS:
        successes = results[task]
        all_successes.extend(successes)
        rate = sum(successes) / len(successes) if successes else 0.0
        print(f"  {task:20s}: {sum(successes):2d}/{len(successes):2d} = {rate:.3f}")

    overall = sum(all_successes) / len(all_successes) if all_successes else 0.0
    print(f"  {'Overall':20s}: {sum(all_successes):2d}/{len(all_successes):2d} = {overall:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tyro.cli(main)
