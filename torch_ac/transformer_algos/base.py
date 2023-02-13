from abc import ABC, abstractmethod
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, preprocess_obss, cfg):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = cfg.frames_per_proc
        self.discount = cfg.discount
        self.lr = cfg.lr
        self.gae_lambda = cfg.gae_lambda
        self.entropy_coef = cfg.entropy_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.recurrence = cfg.recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = cfg.reshape_reward
        self.num_decoder_layers = cfg.num_decoder_layers

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        self.use_pastkv = cfg.use_pastkv
        if self.use_pastkv:
            self.memory = torch.zeros(shape[1], self.recurrence, self.num_decoder_layers, self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.recurrence, self.num_decoder_layers, self.acmodel.memory_size, device=self.device)
        else:
            self.memory = torch.zeros(shape[1], self.recurrence, self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.recurrence, self.acmodel.memory_size, device=self.device)
            self.act_indices = torch.zeros(*shape, dtype=torch.long, device=self.device)
            self.act_ind = torch.zeros(self.num_procs, dtype=torch.long, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self, bc_mode):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            self.obss[i] = self.obs
            if self.use_pastkv:
                preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                with torch.no_grad():
                    if self.acmodel.recurrent:
                        # print("memory shape when collecting experience: ", self.memory.shape)
                        dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1).unsqueeze(1).unsqueeze(1))
                        if memory.shape[1] < self.recurrence:
                            memory = torch.cat((memory, torch.zeros(memory.shape[0], 
                                self.recurrence - memory.shape[1], memory.shape[2], memory.shape[3], device=self.device)), dim=1)
                        else:
                            memory = memory[:, 1:, ...]
                    else:
                        dist, value = self.acmodel(preprocessed_obs)
                action = dist.sample()

                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = tuple(a | b for a, b in zip(terminated, truncated))
            else:
                preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                with torch.no_grad():
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory, self.act_ind)
                action = dist.sample()
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                gt_action = self.env.compute_expert_action()
                done = tuple(a | b for a, b in zip(terminated, truncated))
            # Update experiences values
            self.obs = obs
            if self.use_pastkv:
                # store the input memory of current frame
                self.memories[i] = self.memory
                self.memory = memory
            else:
                # store the observation of current frame
                self.memories[i] = self.memory
                self.memory = torch.cat((self.memory[:, 1:], memory), dim=1)
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.act_indices[i] = self.act_ind
            self.act_ind = self.act_ind + 1
            self.act_ind = (self.act_ind * self.mask).long()
            self.act_ind[self.act_ind > self.recurrence-1] = self.recurrence-1
            if bc_mode:
                self.actions[i] = torch.tensor(gt_action, device=self.device, dtype=torch.int)
            else:
                self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.use_pastkv:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1).unsqueeze(1).unsqueeze(1))
            else:
               _, next_value, _ = self.acmodel(preprocessed_obs, self.memory, self.act_ind)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        if not self.use_pastkv:
            exps.act_indices = self.act_indices.transpose(0, 1).reshape(-1)
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
