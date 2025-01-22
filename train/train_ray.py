# this code is the editted implementaiton of the blog below.
# Reference: https://medium.com/@kaige.yang0110/ray-rllib-how-to-train-dreamerv3-on-vizdoom-and-atari-122c8bd1170b

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
import skimage.transform
from utils import ObservationWrapper

import vizdoom.gymnasium_wrapper  # noqa
import vizdoom
from vizdoom.gymnasium_wrapper.gymnasium_env_defns import VizdoomScenarioEnv
from ray.tune.registry import register_env
from utils import wrap_env, reward_wrap_env

import ray
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config

import tqdm


def main(train_config):

    # config 

    config = {"scenario_file": "basic.cfg"}
    def env_creator(env_config):
        return wrap_env(VizdoomScenarioEnv(**config), train_config)
    register_env('vizdoom_env', env_creator)

    # Create DreamerV3

    num_cpus = int(ray.cluster_resources()['CPU'])
    num_gpus = int(ray.cluster_resources()['GPU'])

    num_learner_workers = num_gpus-1
    num_gpus_per_learner_worker = 1
    num_cpus_per_learner_workers = 1

    config = (
            DreamerV3Config()
            .environment(
                env='vizdoom_env',
            )
            .resources(
                num_learner_workers=num_learner_workers,
                num_gpus_per_learner_worker=1,
                # num_cpus_for_local_worker=1,
                num_cpus_per_learner_worker=num_cpus_per_learner_workers,
            )
            .rollouts(num_envs_per_worker=1, remote_worker_envs=False)
            .training(
                model_size="S",
                training_ratio=512,
                batch_size_B=16*num_learner_workers,
            )

        )

    # run training
    iteration_num = train_config['ray_iterations']

    algo = config.build()
    print('------ algo=', algo)
    for iteration in tqdm(range(iteration_num)):
        result = algo.train()
        print('result.keys', result.keys())


    # shutdown ray
    ray.shutdown()

if __name__ == "__main__":
    main()