# Disclaimer
# This code is adapted from DreamerV3 repository by Hafner et al.:
# https://github.com/danijar/dreamerv3
# To reproduce our training pipeline, overwrite dreamerv3/embodied/envs/crafter.py with this file.
# Original: https://github.com/danijar/dreamerv3/blob/main/embodied/envs/crafter.py

# What's Modified: added a feature that write observation tensor and player position vector every 100 steps
#   1. to perform probing later with the images and positions
#   See "editted" comments for our modifications


import json

import crafter
import elements
import embodied
import numpy as np


class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), logs=False, logdir=None, seed=None):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True
    self._total_step = 0

  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._env.observation_space.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
    }
    if self._logs:
      spaces.update({
          f'log/achievement_{k}': elements.Space(np.int32)
          for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    
    # image: np.array <-> obs = _obs() = render() in crafter/crafter/env.py
    # defalt size = (64, 64, 3)
    image, reward, self._done, info = self._env.step(action['action'])
    self._reward += reward
    self._length += 1
    self._total_step += 1

    # editted
    if self._logdir and self._total_step%100 == 0:
      self._write_stats(self._length, self._reward, info, image, self._total_step)

    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{'log/reward': np.float32(info['reward'] if info else 0.0)},
    )
    if self._logs:
      log_achievements = {
          f'log/achievement_{k}': info['achievements'][k] if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
    return obs

  # editted
  def _write_stats(self, length, reward, info, image, step):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
        
        # editted
        # player_pos: np.array
        # image: np.array
        'step': step,
        'pos': info['player_pos'].tolist(),
        'image': image.tolist(),
    }

    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines, mode='w')
    # editted
    print(f'Wrote stats at total step {step}: {filename}', end="\r")
