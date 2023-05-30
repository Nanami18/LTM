import argparse
import time
import datetime
import torch_ac
import torch
from torch.utils.data import DataLoader
import numpy as np
import tensorboardX
import sys

import utils
from utils import device
from configs.config import cfg, cfg_from_file

import envs
from models.decision_transformer_model import build_model
from decision_transformer.dataset_generation import generate_expert_trajectories, generate_random_trajectories, HallwayMemoryEnvDataset


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--use_expert", action="store_true")
parser.add_argument("--custom_dir", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set seed for all randomness sources

    cfg_from_file(args.config)
    utils.seed(cfg.seed)

    # Set device

    print(f"Device: {device}\n")

    # Create env
    env = utils.make_env(cfg.env_name, cfg.seed, render_mode='human')
    env.render()

    if args.custom_dir is not None:
        model_name = args.custom_dir
    else:
        model_name = str(args.config).split("/")[-1][:-5]
    model_dir = utils.get_model_dir(model_name, cfg.env_name)

    model = build_model(cfg, env.observation_space, env.action_space)
    model.to(device)

    # Load training status

    status = utils.get_status(model_dir)
    model.load_state_dict(status["model_state"])
    model.eval()
    print("Model loaded\n")

    
    episodes_num = 0
    cum_episode_return = 0
    
    with torch.no_grad():
        
        while episodes_num < cfg.episodes:
            
            terminated = False
            truncated = False
            cur_episode_return = 0
            obs, _ = env.reset()
            states = model.convert_obs_inf(obs)
            states = states.to(device)
            timesteps = torch.zeros((1,1), device=device).long()
            rewards = torch.tensor(cfg.eval_reward, device=device).float()
            rewards = rewards.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            actions = None
            counter = 0
            memory = None
            
            while not terminated and not truncated:
                if cfg.use_rmt:
                    action_logits, memory = model.inference(rewards, states, actions, timesteps, None, memory)
                else:
                    action_logits = model.inference(rewards, states, actions, timesteps)
                if cfg.argmax:
                    action = torch.argmax(action_logits, dim=2)
                else:
                    action = torch.distributions.Categorical(logits=action_logits).sample()
                
                obs, reward, terminated, truncated, _ = env.step(action[:, -1].item())
                cur_episode_return += reward
                counter += 1
                if counter >= cfg.max_timestep-1:
                    truncated = True
                
                # Concatenate the new state, action, reward, and timestep
                if states.shape[1] >= cfg.context_length:
                    states = states[:,1:]
                    rewards = rewards[:,1:]
                    actions = actions[:,1:]
                    timesteps = timesteps[:,1:]
                states = torch.cat((states,model.convert_obs_inf(obs).to(device)), dim=1)
                rewards = torch.cat((rewards, rewards[:,-1]-torch.tensor(reward, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)), dim=1)
                # if actions is not None:
                #     actions = torch.cat((actions, action[:, -1]), dim=1)
                # else:
                actions = action
                timesteps = torch.cat((timesteps, torch.tensor(counter, device=device).unsqueeze(0).unsqueeze(0)), dim=1)
                time.sleep(cfg.pause)

            print(f"Episode {episodes_num} return: {cur_episode_return}")
            cum_episode_return += cur_episode_return
            episodes_num += 1

            if env.window.closed:
                break

        print(f"Average return: {cum_episode_return / episodes_num}")