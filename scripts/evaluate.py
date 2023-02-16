import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device
from configs.config import cfg, cfg_from_file

from envs.memory_minigrid import register_envs
register_envs()


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

    # Load environments

    envs = []
    for i in range(cfg.procs):
        env = utils.make_env(cfg.env_name, cfg.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    if args.custom_dir is not None:
        model_name = args.custom_dir
    else:
        model_name = str(args.config).split("/")[-1][:-5] or default_model_name
    model_dir = utils.get_model_dir(model_name)
    if cfg.use_lstm:
        agent = utils.Agent(env.observation_space, env.action_space, model_dir, cfg=cfg)
    else:
        agent = utils.TransformerAgent(env.observation_space, env.action_space, model_dir, cfg=cfg)
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(cfg.procs, device=device)
    log_episode_num_frames = torch.zeros(cfg.procs, device=device)

    while log_done_counter < cfg.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(cfg.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    # Print worst episodes

    n = cfg.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
