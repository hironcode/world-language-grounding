import gymnasium as gym
from vizdoom import gymnasium_wrapper
import numpy as np
import torch
from collections import deque
import utils

from dreamerv3.dreamerv3.agent import Agent as DreamerV3

ENV_NAME = "VizdoomBasic-v0"

env = gym.make(ENV_NAME)

train_config = utils.load_config("./config.json")['training']

agent = DreamerV3()

obs, info = env.reset()
for _ in range(1000):
   action = agent(observation)  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()