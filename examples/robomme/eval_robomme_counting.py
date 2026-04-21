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

    # MEM short-term memory. Must match the training config for the checkpoint
    # being evaluated (see `pi05_robomme_counting_lora`).
    num_memory_frames: int = 6
    memory_stride_seconds: float = 1.0


def _single_frame_obs(obs: dict) -> dict:
    front_rgb = np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8)
    wrist_rgb = np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8)
    joint_state = np.asarray(obs["joint_state_list"][-1], dtype=np.float32)
    gripper_state = np.asarray(obs["gripper_state_list"][-1], dtype=np.float32)
    if gripper_state.shape[-1] == 2:
        gripper_state = gripper_state[..., :1]
    return {
        "front_rgb": front_rgb,
        "wrist_rgb": wrist_rgb,
        "joint_state": joint_state,
        "gripper_state": gripper_state,
    }


class MemoryBuffer:
    """Client-side history for MEM short-term memory inference.

    Stores every per-step observation and, on `snapshot`, emits K frames sampled
    at `stride_steps` intervals ending at the most recent observation. The
    earliest slots are padded with the first recorded observation when the
    episode is shorter than `(num_frames - 1) * stride_steps + 1`, matching how
    LeRobot's `delta_timestamps` handles episode-start padding during training.
    """

    def __init__(self, num_frames: int, stride_steps: int):
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if stride_steps < 1:
            raise ValueError(f"stride_steps must be >= 1, got {stride_steps}")
        self.num_frames = num_frames
        self.stride_steps = stride_steps
        self._history: list[dict] = []

    def append(self, frame: dict) -> None:
        self._history.append(frame)

    def _frame_at(self, step_idx: int) -> dict:
        clamped = max(step_idx, 0)
        return self._history[clamped]

    def snapshot(self) -> dict:
        if not self._history:
            raise RuntimeError("MemoryBuffer is empty; call append() before snapshot().")
        latest = len(self._history) - 1
        frames = [self._frame_at(latest - (self.num_frames - 1 - i) * self.stride_steps) for i in range(self.num_frames)]
        if self.num_frames == 1:
            return frames[0]
        return {key: np.stack([f[key] for f in frames], axis=0) for key in frames[0]}


def _policy_obs_from_snapshot(snapshot: dict, task_goal: str) -> dict:
    return {
        "observation/front_rgb": snapshot["front_rgb"],
        "observation/wrist_rgb": snapshot["wrist_rgb"],
        "observation/joint_state": snapshot["joint_state"],
        "observation/gripper_state": snapshot["gripper_state"],
        "prompt": task_goal,
    }


def _grab_frame(obs: dict) -> np.ndarray:
    front = np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8)
    wrist = np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8)
    return np.hstack([front, wrist])


def main(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    stride_steps = max(int(round(args.memory_stride_seconds * args.video_fps)), 1)
    logging.info(
        "Memory buffer: K=%s frames, stride=%.2fs (%s steps at %sfps)",
        args.num_memory_frames,
        args.memory_stride_seconds,
        stride_steps,
        args.video_fps,
    )

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

            buffer = MemoryBuffer(args.num_memory_frames, stride_steps)
            buffer.append(_single_frame_obs(obs))

            frames: list[np.ndarray] = []
            if args.save_videos:
                frames.append(_grab_frame(obs))

            action_plan: collections.deque = collections.deque()
            outcome = "unknown"
            done = False

            while not done:
                if not action_plan:
                    policy_obs = _policy_obs_from_snapshot(buffer.snapshot(), task_goal)
                    action_chunk = client.infer(policy_obs)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, _reward, terminated, truncated, info = env.step(action)
                buffer.append(_single_frame_obs(obs))

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
