from collections import deque

import gym
import gymnasium
import numpy as np

from envs.wrappers.timeout import Timeout


MEMORY_MAZE_TASKS = {
	"memory-maze-9x9":   ("memory_maze:MemoryMaze-9x9-v0",   1000),
	"memory-maze-11x11": ("memory_maze:MemoryMaze-11x11-v0", 2000),
	"memory-maze-13x13": ("memory_maze:MemoryMaze-13x13-v0", 3000),
	"memory-maze-15x15": ("memory_maze:MemoryMaze-15x15-v0", 4000),
}

class MemoryMazeWrapper:
	def __init__(self, env, num_frames=3):
		self.env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)
		self._last_raw_obs = None
		self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
		self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(num_frames * 3, 64, 64), dtype=np.uint8)

	@property
	def unwrapped(self):
		return self.env

	def _get_obs(self, raw_obs):
		frame = np.transpose(raw_obs, (2, 0, 1))
		self._frames.append(frame)
		return np.concatenate(list(self._frames), axis=0)

	def reset(self, **kwargs):
		raw_obs = self.env.reset()
		self._last_raw_obs = raw_obs
		frame = np.transpose(raw_obs, (2, 0, 1))
		for _ in range(self._num_frames):
			self._frames.append(frame)
		return np.concatenate(list(self._frames), axis=0)

	def step(self, action):
		discrete_action = int(np.argmax(action))
		result = self.env.step(discrete_action)
		if len(result) == 5:
			raw_obs, reward, terminated, truncated, info = result
			done = terminated or truncated
		else:
			raw_obs, reward, done, info = result
		self._last_raw_obs = raw_obs
		obs = self._get_obs(raw_obs)
		return obs, reward, done, info

	def render(self, **kwargs):
		return self._last_raw_obs


def make_env(cfg):
	if cfg.task not in MEMORY_MAZE_TASKS:
		raise ValueError(f"Unknown Memory Maze task: {cfg.task}")

	gym_id, max_steps = MEMORY_MAZE_TASKS[cfg.task]
	env = gym.make(gym_id)
	cfg.obs = "rgb"
	env = MemoryMazeWrapper(env)
	env = Timeout(env, max_episode_steps=max_steps)
	return env
