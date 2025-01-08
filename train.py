import gymnasium as gym
from vizdoom import gymnasium_wrapper
import numpy as np
import torch
from collections import deque

from dreamerv3.dreamerv3 import agent


env = gym.make("VizdoomDeadlyCorridor-v0")

agent = 

obs, info = env.reset()
for _ in range(1000):
   action = policy(observation)  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()