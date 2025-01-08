# ObservationWrapper, wrap_env, and reward_wrap_env are the implementation of a blog below 
# Reference: https://medium.com/@kaige.yang0110/ray-rllib-how-to-train-dreamerv3-on-vizdoom-and-atari-122c8bd1170b

import gym
import cv2
import numpy as np
import json

IMAGE_SHAPE = (64, 64)
FRAME_SKIP = 4

class ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env, shape=IMAGE_SHAPE, frame_skip=FRAME_SKIP):
        super().__init__(env)
        self.image_shape = shape
        # print('shape', shape)
        self.image_shape_reverse = shape[::-1]
        # print('image_shape_reverse', self.image_shape_reverse)
        self.env.frame_skip = frame_skip

        # Create new observation space with the new shape
        # print('env.obs', env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (self.image_shape[0], self.image_shape[1], num_channels)
        # print('new_shape', new_shape)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        # print('observation["screen"].shape', observation["screen"].shape)
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        # print('obs.shape', observation.shape)
        observation = observation.astype('float32')
        # print('obs.shape', observation.shape)
        return observation

def wrap_env(env, config):
    env = ObservationWrapper(env, shape=config['image_shape'], frame_skip=config['frame_skip'])
    env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
    return env

def reward_wrap_env(env):
    env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
    return env

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config