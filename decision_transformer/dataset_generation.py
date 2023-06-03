import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

from minigrid.core.actions import Actions

# Generate a dataset of expert demonstrations, stored as an tuple of (s,a,r) arrays
def generate_expert_trajectories(env, episodes, reward_pertubation=0.0):
    
    data = []
    for i in range(episodes):
        cur_obs = []
        cur_acts = []
        cur_rewards = []
        terminated = False
        truncated = False

        obs, _ = env.reset()
        cur_obs.append(obs)
        while not terminated and not truncated:
            action = env.compute_expert_action()
            cur_acts.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            cur_rewards.append(reward)
            if not terminated:
                cur_obs.append(obs)
        
        data.append((cur_obs, cur_acts, cur_rewards))

    return data

def generate_random_trajectories(env, episodes):
    
    data = []
    for i in range(episodes):
        cur_obs = []
        cur_acts = []
        cur_rewards = []
        terminated = False
        truncated = False

        obs, _ = env.reset()
        cur_obs.append(obs)
        while not terminated and not truncated:
            action = env.compute_random_action()
            cur_acts.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            cur_rewards.append(reward)
            if not terminated:
                cur_obs.append(obs)
        
        data.append((cur_obs, cur_acts, cur_rewards))

    return data

class HallwayMemoryEnvDataset(Dataset):
    # Trajectories is supposed to be list of episodes, each element is a tuple of (s,a,r) arrays
    def __init__(self, trajectories, cfg):
        self.trajectories = trajectories
        self.num_traj = len(trajectories)
        self.context_length = cfg.context_length if not cfg.use_rmt else cfg.context_length * (cfg.num_bptt+1)
        self.gamma = cfg.discount
        self.scalar_only = "scalarobs" in cfg.env_name
        self.sampling_from_beginning = cfg.sampling_from_beginning
        if not self.scalar_only:
            self.image_obs_dim = trajectories[0][0][0]['image'].shape
        else:
            self.image_obs_dim = 1
        self.act_dim = 1
    
    def __len__(self):
        return len(self.trajectories)
    
    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(len(x)-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        
        return discount_cumsum
    
    def __getitem__(self, idx):
        cur_traj = self.trajectories[idx]
        # Notes: This design choice is specific for hallway memory environment, as the first observation is necessary to make the correct decision
        if self.sampling_from_beginning:
            start = 0
        else:
            start = np.random.randint(0, len(cur_traj[0])-1)
        end = min(start + self.context_length, len(cur_traj[0]))
        if not self.scalar_only:
            states = [obs['image'] for obs in cur_traj[0][start:end]]
        else:
            states = [obs for obs in cur_traj[0][start:end]]
        actions = cur_traj[1][start:end]
        rewards = cur_traj[2][start:end]
        rewards = self._discount_cumsum(rewards, self.gamma)
        if self.act_dim == 1:
            actions = np.expand_dims(actions, axis=-1)
        rewards = np.expand_dims(rewards, axis=-1)

        # Pad to right
        # if not self.scalar_only:
        #     states = np.concatenate([np.zeros((self.context_length - len(states), *self.image_obs_dim)), states], axis=0)
        # else:
        #     states = np.concatenate([np.zeros((self.context_length - len(states))), states], axis=0)
        # actions = np.concatenate([actions, np.zeros((self.context_length - len(actions), self.act_dim))], axis=0)
        # rewards = np.concatenate([rewards, np.zeros((self.context_length - len(rewards), 1))], axis=0)
        # timesteps = np.concatenate((np.arange(start, end), np.zeros(self.context_length - (end-start))), axis=0)
        
        # mask == 1 `states` is valid, mask == 0 `states` is invalid (padding)
        # mask = np.concatenate((np.ones(end-start), np.zeros(self.context_length - (end-start))), axis=0)
        # mask = np.repeat(mask, 3)

        # Pad to left
        if not self.scalar_only:
            states = np.concatenate([np.zeros((self.context_length - len(states), *self.image_obs_dim)), states], axis=0)
        else:
            states = np.concatenate([np.zeros((self.context_length - len(states))), states], axis=0)
        actions = np.concatenate([np.zeros((self.context_length - len(actions), self.act_dim)), actions], axis=0)
        rewards = np.concatenate([np.zeros((self.context_length - len(rewards), 1)), rewards], axis=0)
        timesteps = np.concatenate((np.zeros(self.context_length - (end-start)), np.arange(start, end)), axis=0)

        # mask == 1 `states` is valid, mask == 0 `states` is invalid (padding)
        mask = np.concatenate((np.zeros(self.context_length - (end-start)), np.ones(end-start)), axis=0)
        mask = np.repeat(mask, 3)

        # Convert attention to mask to be 2D, with an extra dimension for attention heads
        mask = torch.tensor(mask)
        T = mask.shape[0]
        mask = mask.view(1, 1, T)
        mask = mask.repeat(1, T, 1)

        # Convert to torch tensors
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        timesteps = torch.from_numpy(timesteps).long()

        return states, actions, rewards, timesteps, mask

class FindingObjEnvDataset(Dataset):
    # Trajectories is supposed to be list of episodes, each element is a tuple of (s,a,r) arrays
    def __init__(self, trajectories, cfg):
        self.trajectories = trajectories
        self.num_traj = len(trajectories)
        self.context_length = cfg.context_length if not cfg.use_rmt else cfg.context_length * (cfg.num_bptt+1)
        self.gamma = cfg.discount
        self.scalar_only = "scalarobs" in cfg.env_name
        self.sampling_from_beginning = cfg.sampling_from_beginning
        if not self.scalar_only:
            self.image_obs_dim = trajectories[0][0][0]['image'].shape
        else:
            self.image_obs_dim = 1
        self.act_dim = 1
    
    def __len__(self):
        return len(self.trajectories)
    
    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x, dtype=np.float32)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(len(x)-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum
    
    def __getitem__(self, idx):
        cur_traj = self.trajectories[idx]
        # Notes: This design choice is specific for hallway memory environment, as the first observation is necessary to make the correct decision
        start = np.random.randint(0, len(cur_traj[0])-1)
        end = min(start + self.context_length, len(cur_traj[0]))
        state_imageobs = [obs['image'] for obs in cur_traj[0][start:end]]
        state_target = [obs['target_color'] for obs in cur_traj[0][start:end]]
        actions = cur_traj[1][start:end]
        rewards = cur_traj[2][start:end]
        rewards = self._discount_cumsum(rewards, self.gamma)
        if self.act_dim == 1:
            actions = np.expand_dims(actions, axis=-1)
        rewards = np.expand_dims(rewards, axis=-1)

        # Pad to right
        # if not self.scalar_only:
        #     states = np.concatenate([np.zeros((self.context_length - len(states), *self.image_obs_dim)), states], axis=0)
        # else:
        #     states = np.concatenate([np.zeros((self.context_length - len(states))), states], axis=0)
        # actions = np.concatenate([actions, np.zeros((self.context_length - len(actions), self.act_dim))], axis=0)
        # rewards = np.concatenate([rewards, np.zeros((self.context_length - len(rewards), 1))], axis=0)
        # timesteps = np.concatenate((np.arange(start, end), np.zeros(self.context_length - (end-start))), axis=0)
        
        # mask == 1 `states` is valid, mask == 0 `states` is invalid (padding)
        # mask = np.concatenate((np.ones(end-start), np.zeros(self.context_length - (end-start))), axis=0)
        # mask = np.repeat(mask, 3)

        # Pad to left
        state_imageobs = np.concatenate([np.zeros((self.context_length - len(state_imageobs), *self.image_obs_dim)), state_imageobs], axis=0)
        state_target = np.concatenate([np.zeros(self.context_length - len(state_target)), state_target], axis=0)
        actions = np.concatenate([np.zeros((self.context_length - len(actions), self.act_dim)), actions], axis=0)
        rewards = np.concatenate([np.zeros((self.context_length - len(rewards), 1)), rewards], axis=0)
        timesteps = np.concatenate((np.zeros(self.context_length - (end-start)), np.arange(start, end)), axis=0)

        # mask == 1 `states` is valid, mask == 0 `states` is invalid (padding)
        mask = np.concatenate((np.zeros(self.context_length - (end-start)), np.ones(end-start)), axis=0)
        mask = np.repeat(mask, 3)

        # Convert attention to mask to be 2D, with an extra dimension for attention heads
        mask = torch.tensor(mask)
        T = mask.shape[0]
        mask = mask.view(1, 1, T)
        mask = mask.repeat(1, T, 1)

        # Convert to torch tensors
        states = {'image' : torch.from_numpy(state_imageobs).float(), 'target' : torch.from_numpy(state_target).float()}
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        timesteps = torch.from_numpy(timesteps).long()

        return states, actions, rewards, timesteps, mask

def collate_fn_findingobj(batches):
    states, actions, rewards, timesteps, mask = [], [], [], [], []
    for batch in batches:
        states.append(batch[0])
        actions.append(batch[1])
        rewards.append(batch[2])
        timesteps.append(batch[3])
        mask.append(batch[4])
    
    states = {k: torch.stack([d[k] for d in states]) for k in states[0].keys()}
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    timesteps = torch.stack(timesteps)
    mask = torch.stack(mask)

    return states, actions, rewards, timesteps, mask


if __name__ == "__main__":
    import argparse
    import envs
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    env = gym.make("MiniGrid-MemoryS9-scalarobs", render_mode=None)
    trajectories = generate_expert_trajectories(env, args.episodes)
    data = HallwayMemoryEnvDataset(trajectories, 10, 0.99, True)
