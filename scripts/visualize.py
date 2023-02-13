import argparse
import numpy

import utils
from utils import device
import time

import json
import argparse
import yaml
from easydict import EasyDict as edict
import os
import numpy as np


from envs.memory_minigrid import register_envs
register_envs()

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--use_expert", action="store_true")
args = parser.parse_args()
# Set seed for all randomness sources

with open(args.config, "r") as f:
    cfg = edict(yaml.safe_load(f))
utils.seed(cfg.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(cfg.env_name, cfg.seed, render_mode="human")
for _ in range(cfg.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(str(args.config).split("/")[-1][:-5])
if not cfg:
    agent = utils.Agent(env.observation_space, env.action_space, model_dir, cfg=cfg)
else:
    agent = utils.TransformerAgent(env.observation_space, env.action_space, model_dir, cfg=cfg)
print("Agent loaded\n")

# Run the agent

if cfg.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

for episode in range(cfg.episodes):
    obs, _ = env.reset()

    print("episode starts")
    num_acts = 0
    while True:
        env.render()
        if cfg.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
        
        if args.use_expert and num_acts > 15:
            action = env.compute_expert_action()
        else:
            action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)
        num_acts += 1
        time.sleep(cfg.pause)

        if done or env.window.closed:
            break
    
    print("episode ends, num actions executed: ", num_acts)
    print("reward: ", reward)
    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
