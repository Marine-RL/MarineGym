import logging
import os
import time

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import OmegaConf

from marinegym import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from marinegym.utils.torchrl import SyncDataCollector
from marinegym.utils.torchrl.transforms import (
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    # VelController,
    AttitudeController,
    RateController,
    History
)
from marinegym.utils.wandb import init_wandb
from marinegym.utils.torchrl import RenderCallback, EpisodeStats
from marinegym.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


def load_checkpoint(checkpoint_path, env_config, algo_config):
    from marinegym.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[env_config.task.name]
    base_env = env_class(env_config, headless=env_config.headless)
    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    policy = ALGOS[algo_config.name.lower()](
        algo_config, 
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=base_env.device
    )
    policy.load_state_dict(torch.load(checkpoint_path))
    return policy, env



# Evaluate the loaded model
def evaluate_model(env, policy, num_episodes, cfg):
    from torchrl.envs.utils import set_exploration_type, ExplorationType

    from marinegym.sensors.camera import Camera, PinholeCameraCfg
    from torchvision.io import write_video
    import dataclasses

    sim_dt = cfg.sim.dt

    results = []
    frames_vis = np.empty((0,cfg.viewer.resolution[1],cfg.viewer.resolution[0],3))
    env.eval()
    env.set_seed(0)
    env.enable_render(True)
    render_callback = RenderCallback(interval=1)
    for i in tqdm(range(num_episodes)):
        with set_exploration_type(ExplorationType.MODE):
            traj = env.rollout(
                max_steps=env.base_env.max_episode_length,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False
            )
        results.append(traj["next", "stats"].cpu())
    return results

FILE_PATH = os.path.dirname(__file__)
@hydra.main(config_path=FILE_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))
    policy, env = load_checkpoint(hover_lauv_pure, cfg, cfg.algo)
    eval_results = evaluate_model(env, policy, num_episodes=100, cfg=cfg)
    print(eval_results)

if __name__ == "__main__":
    main()