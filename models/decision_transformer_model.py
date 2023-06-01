import collections
import math
import random

import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    DIR_TO_VEC
)

from minigrid.core.actions import Actions

def build_model(cfg, obs_space, action_space):
    if "Memory" in cfg.env_name:
        if "scalarobs" in cfg.env_name:
            if cfg.use_rmt:
                return HallwayMemoryScalarObs_DTwithRMT(cfg, obs_space, action_space)
            else:
                return HallwayMemoryScalarObs_DT(cfg, obs_space, action_space)
        else:
            return HallwayMemory_DT(cfg, obs_space, action_space)

def init_params(m, cfg):
    classname = m.__class__.__name__
    if (classname.find("Linear") != -1 and classname.find("ACModel") == -1) or classname.find("Embedding") != -1 :
        m.weight.data.normal_(0, 1)
        if cfg.init_var:
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if classname != "Embedding" and m.bias is not None:
            m.bias.data.fill_(0)

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        if attention_mask is not None:
            # Repeat the attention mask for each head and each element
            # mask_edit = attention_mask.view(B, 1, 1, T)
            # mask_edit = mask_edit.repeat(1, 1, T, 1)
            # Preserve the causal mask while masking out the padding
            # mask = self.mask * attention_mask
            # indices = torch.arange(T)
            # Always allow self attention
            # mask = attention_mask
            # mask[:, :, indices, indices] = 1
            weights = weights.masked_fill(attention_mask[...,:T,:T] == 0, float('-inf'))
        else:
            weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, attention_mask=None):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x, attention_mask) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x

class SequentialTransformerBlock(nn.Sequential):
    def forward(self, x, attention_mask=None):
        for module in self._modules.values():
            x = module(x, attention_mask)
        return x

# The DT implementation for spatial observation, hard-coded causal attention
class HallwayMemory_DT(nn.Module):
    def __init__(self, cfg, observation_space, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.hidden_dim = cfg.hidden_dim

        input_seq_len = 3 * cfg.context_length
        self.max_length = input_seq_len
        blocks = [Block(cfg.hidden_dim, input_seq_len, cfg.n_heads, cfg.drop_p) for _ in range(cfg.n_blocks)]
        self.transformer = SequentialTransformerBlock(*blocks)

        # input encoders
        self.embed_timestep = nn.Embedding(cfg.max_timestep, self.hidden_dim) # shall we use positional encoding?
        self.embed_return = nn.Linear(self.action_dim, self.hidden_dim)
        # Discrete action needs to have shape B x T, where continuous action has shape B x T x A
        if cfg.discrete_action:
            self.embed_action = nn.Embedding(len(Actions)+1, self.hidden_dim)
        else:
            self.embed_action = nn.Linear(cfg.action_dim, self.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.hidden_dim)
        # State embedding
        if cfg.use_linear_state_encoder:
            self.state_embed = nn.Linear(np.prod(observation_space['image'].shape), self.hidden_dim)
        else:
            self.object_embed = nn.Embedding(len(OBJECT_TO_IDX), cfg.token_embed_size)
            self.color_embed = nn.Embedding(len(COLOR_TO_IDX), cfg.token_embed_size)
            self.state_embed = nn.Embedding(2, cfg.token_embed_size)
            
            self.image_conv = nn.Sequential(
                nn.Conv2d(cfg.token_embed_size*3, cfg.image_embed_size//2, (2, 2)),
                nn.ReLU(),
                nn.AvgPool2d((2, 2)),
                nn.Conv2d(cfg.image_embed_size//2, cfg.image_embed_size//2, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(cfg.image_embed_size//2, cfg.image_embed_size, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(cfg.image_embed_size, cfg.hidden_dim),
                nn.ReLU(),
            )

        # predictors
        self.action_predictor = nn.Linear(self.hidden_dim, len(Actions))
        # self.state_predictor = nn.Linear(self.hidden_dim, self.state_dim)
        # self.return_predictor = nn.Linear(self.hidden_dim, 1)

        # Attention mask related
        ones = torch.ones((self.max_length, self.max_length))
        self.causal_mask = torch.tril(ones).view(1, 1, self.max_length, self.max_length).cuda()

    def forward(self, returns, states, actions, timestep, attention_mask=None):

        B, T = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timestep)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # Embed states
        if self.cfg.use_linear_state_encoder:
            states_embeddings = self.state_embed(states.reshape(B*T, -1).float())
            states_embeddings = states_embeddings.reshape(B, T, self.hidden_dim) + time_embeddings
        else:
            object_embeddings = self.object_embed(states[:,:,:,:,0].long())
            color_embeddings = self.color_embed(states[:,:,:,:,1].long())
            state_embeddings = self.state_embed(states[:,:,:,:,2].long())
            states_embeddings = torch.cat([object_embeddings, color_embeddings, state_embeddings], dim=-1)
            states_embeddings = states_embeddings.reshape(B*T, states_embeddings.shape[2], states_embeddings.shape[3], states_embeddings.shape[4])
            states_embeddings = states_embeddings.permute(0,3,1,2)
            states_embeddings = self.image_conv(states_embeddings)
            states_embeddings = states_embeddings.reshape(B, T, self.hidden_dim) + time_embeddings

        if actions is not None:
            actions_embeddings = torch.squeeze(self.embed_action(actions), 2) + time_embeddings

        if actions is not None:
            hidden = torch.stack([returns_embeddings, states_embeddings, actions_embeddings], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.hidden_dim)
        else:
            hidden = torch.stack([returns_embeddings, states_embeddings], dim=1).permute(0,2,1,3).reshape(B, 2*T, self.hidden_dim)
        hidden = self.embed_ln(hidden)

        # Transformer processing
        ones = torch.ones((self.max_length, self.max_length))
        causal_mask = torch.tril(ones).view(B, 1, self.max_length, self.max_length).cuda()
        if attention_mask is not None:
            attention_mask = causal_mask * attention_mask
            indices = torch.arange(T)
            attention_mask[:, :, indices, indices] = 1
        hidden = self.transformer(hidden, attention_mask)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        if actions is not None:
            hidden = hidden.reshape(B, T, 3, self.hidden_dim).permute(0,2,1,3)
        else:
            hidden = hidden.reshape(B, T, 2, self.hidden_dim).permute(0,2,1,3)

        action_logits = self.action_predictor(hidden[:,1])

        return action_logits

    # batch dimension always equal to 1 right now, will probably remove it in the future
    # Actions will be one timestep less, as we don't know the future action
    def inference(self, returns, states, actions, timesteps, attention_mask=None):

        B, T = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timesteps)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # Embed states
        if self.cfg.use_linear_state_encoder:
            states_embeddings = self.state_embed(states.reshape(B*T, -1).float())
            states_embeddings = states_embeddings.reshape(B, T, self.hidden_dim) + time_embeddings
        else:
            object_embeddings = self.object_embed(states[:,:,:,:,0].long())
            color_embeddings = self.color_embed(states[:,:,:,:,1].long())
            state_embeddings = self.state_embed(states[:,:,:,:,2].long())
            states_embeddings = torch.cat([object_embeddings, color_embeddings, state_embeddings], dim=-1)
            states_embeddings = states_embeddings.reshape(B*T, states_embeddings.shape[2], states_embeddings.shape[3], states_embeddings.shape[4])
            states_embeddings = states_embeddings.permute(0,3,1,2)
            states_embeddings = self.image_conv(states_embeddings)
            states_embeddings = states_embeddings.reshape(B, T, self.hidden_dim) + time_embeddings

        if actions is not None:
            actions = torch.cat([actions, torch.zeros(B, 1).to(actions.device).long()], dim=1)
            if actions.shape[1] > T:
                actions = actions[:,1:]
            actions_embeddings = torch.squeeze(self.embed_action(actions), 2) + time_embeddings

        if actions is not None:
            hidden = torch.stack([returns_embeddings, states_embeddings, actions_embeddings], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.hidden_dim)
        else:
            hidden = torch.stack([returns_embeddings, states_embeddings], dim=1).permute(0,2,1,3).reshape(B, 2*T, self.hidden_dim)
        hidden = self.embed_ln(hidden)

        if attention_mask is not None:
            attention_mask = self.causal_mask * attention_mask
            indices = torch.arange(T)
            attention_mask[:, :, indices, indices] = 1
        hidden = self.transformer(hidden, attention_mask)
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        if actions is not None:
            hidden = hidden.reshape(B, T, 3, self.hidden_dim).permute(0,2,1,3)
        else:
            hidden = hidden.reshape(B, T, 2, self.hidden_dim).permute(0,2,1,3)

        action_logits = self.action_predictor(hidden[:,1])

        return action_logits

    def convert_obs_inf(self, obs):
        states = torch.tensor(obs['image']).unsqueeze(0) # time dimension
        states = states.unsqueeze(0) # batch dimension

        return states


class HallwayMemoryScalarObs_DT(nn.Module):
    def __init__(self, cfg, observation_space, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.hidden_dim = cfg.hidden_dim

        input_seq_len = 3 * cfg.context_length
        self.max_length = input_seq_len
        blocks = [Block(cfg.hidden_dim, input_seq_len, cfg.n_heads, cfg.drop_p) for _ in range(cfg.n_blocks)]
        self.transformer = SequentialTransformerBlock(*blocks)

        # input encoders
        self.embed_timestep = nn.Embedding(cfg.max_timestep, self.hidden_dim) # shall we use positional encoding?
        self.embed_return = nn.Linear(self.action_dim, self.hidden_dim)
        # Discrete action needs to have shape B x T, where continuous action has shape B x T x A
        if cfg.discrete_action:
            self.embed_action = nn.Embedding(len(Actions)+1, self.hidden_dim)
        else:
            self.embed_action = nn.Linear(cfg.action_dim, self.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.hidden_dim)
        # State embedding
        self.state_embed = nn.Embedding(observation_space.n, self.hidden_dim)

        # predictors
        self.action_predictor = nn.Linear(self.hidden_dim, len(Actions))
        # self.state_predictor = nn.Linear(self.hidden_dim, self.state_dim)
        # self.return_predictor = nn.Linear(self.hidden_dim, 1)

        # Attention mask related
        ones = torch.ones((self.max_length, self.max_length))
        self.causal_mask = torch.tril(ones).view(1, 1, self.max_length, self.max_length).cuda()

    def forward(self, returns, states, actions, timestep, attention_mask=None):

        B, T = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timestep)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # Embed states
        states_embeddings = self.state_embed(states.long())

        if actions is not None:
            actions_embeddings = torch.squeeze(self.embed_action(actions), 2) + time_embeddings

        if actions is not None:
            hidden = torch.stack([returns_embeddings, states_embeddings, actions_embeddings], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.hidden_dim)
        else:
            hidden = torch.stack([returns_embeddings, states_embeddings], dim=1).permute(0,2,1,3).reshape(B, 2*T, self.hidden_dim)
        
        hidden = self.embed_ln(hidden)
        # Transformer processing
        if attention_mask is not None:
            attention_mask = self.causal_mask * attention_mask
            indices = torch.arange(T)
            attention_mask[:, :, indices, indices] = 1
        hidden = self.transformer(hidden, attention_mask)
        
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        if actions is not None:
            hidden = hidden.reshape(B, T, 3, self.hidden_dim).permute(0,2,1,3)
        else:
            hidden = hidden.reshape(B, T, 2, self.hidden_dim).permute(0,2,1,3)

        action_logits = self.action_predictor(hidden[:,1])

        return action_logits

    # batch dimension always equal to 1 right now, will probably remove it in the future
    # Actions will be one timestep less, as we don't know the future action
    def inference(self, returns, states, actions, timesteps, attention_mask=None):

        B, T = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timesteps)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # Embed states
        states_embeddings = self.state_embed(states.long())

        if actions is not None:
            actions = torch.cat([actions, torch.zeros(B, 1).to(actions.device).long()], dim=1)
            if actions.shape[1] > T:
                actions = actions[:,1:]
            actions_embeddings = torch.squeeze(self.embed_action(actions), 2) + time_embeddings

        if actions is not None:
            hidden = torch.stack([returns_embeddings, states_embeddings, actions_embeddings], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.hidden_dim)
        else:
            hidden = torch.stack([returns_embeddings, states_embeddings], dim=1).permute(0,2,1,3).reshape(B, 2*T, self.hidden_dim)
        hidden = self.embed_ln(hidden)
        hidden = self.transformer(hidden, attention_mask)
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        if actions is not None:
            hidden = hidden.reshape(B, T, 3, self.hidden_dim).permute(0,2,1,3)
        else:
            hidden = hidden.reshape(B, T, 2, self.hidden_dim).permute(0,2,1,3)

        action_logits = self.action_predictor(hidden[:,1])

        return action_logits

    def convert_obs_inf(self, obs):
        states = torch.tensor(obs)
        states = states.unsqueeze(0).unsqueeze(0)

        return states


class HallwayMemoryScalarObs_DTwithRMT(nn.Module):
    def __init__(self, cfg, observation_space, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.hidden_dim = cfg.hidden_dim

        input_seq_len = 3 * cfg.context_length + cfg.memory_size * 2
        self.max_length = input_seq_len
        blocks = [Block(cfg.hidden_dim, input_seq_len, cfg.n_heads, cfg.drop_p) for _ in range(cfg.n_blocks)]
        self.transformer = SequentialTransformerBlock(*blocks)

        # input encoders
        self.embed_timestep = nn.Embedding(cfg.max_timestep, self.hidden_dim) # shall we use positional encoding?
        self.embed_return = nn.Linear(self.action_dim, self.hidden_dim)
        # Discrete action needs to have shape B x T, where continuous action has shape B x T x A
        if cfg.discrete_action:
            self.embed_action = nn.Embedding(len(Actions)+1, self.hidden_dim)
        else:
            self.embed_action = nn.Linear(cfg.action_dim, self.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.hidden_dim)
        # State embedding
        self.state_embed = nn.Embedding(observation_space.n, self.hidden_dim)

        # predictors
        self.action_predictor = nn.Linear(self.hidden_dim, len(Actions))
        # self.state_predictor = nn.Linear(self.hidden_dim, self.state_dim)
        # self.return_predictor = nn.Linear(self.hidden_dim, 1)

        # Memory tokens
        self.memory_size = cfg.memory_size
        self.memory_tokens = nn.Parameter(torch.randn(self.memory_size, self.hidden_dim))
        nn.init.normal(self.memory_tokens, mean=0, std=0.02)
        self.read_mem_embedding = nn.Parameter(torch.zeros(self.memory_size, self.hidden_dim))
        nn.init.normal(self.read_mem_embedding, mean=0, std=0.02)

        # Attention mask related
        ones = torch.ones((self.max_length, self.max_length))
        self.causal_mask = torch.tril(ones).view(1, 1, self.max_length, self.max_length).cuda()

    def forward(self, returns, states, actions, timestep, attention_mask=None, past_memory=None):

        B, T = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timestep)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # Embed states
        states_embeddings = self.state_embed(states.long())

        if actions is not None:
            actions_embeddings = torch.squeeze(self.embed_action(actions), 2) + time_embeddings

        if actions is not None:
            hidden = torch.stack([returns_embeddings, states_embeddings, actions_embeddings], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.hidden_dim)
        else:
            hidden = torch.stack([returns_embeddings, states_embeddings], dim=1).permute(0,2,1,3).reshape(B, 2*T, self.hidden_dim)
        info_T = hidden.shape[1]
        # Handle memory tokens
        if past_memory is None:
            hidden = torch.cat([self.read_mem_embedding.repeat(B, 1, 1), hidden, self.memory_tokens.repeat(B, 1, 1)], dim=1)
        else:
            read_tokens = past_memory + self.read_mem_embedding.repeat(B, 1, 1)
            hidden = torch.cat([read_tokens, hidden, past_memory], dim=1)

        hidden = self.embed_ln(hidden)
        # Transformer processing
        if attention_mask is not None:
            attention_mask = self.causal_mask * attention_mask
            # Always allow self attention to avoid error
            indices = torch.arange(attention_mask.shape[2])
            attention_mask[:, :, indices, indices] = 1
            attention_mask = F.pad(attention_mask, (0, self.memory_size, self.memory_size, 0), value = False)
            attention_mask = F.pad(attention_mask, (self.memory_size, 0, 0, self.memory_size), value = True)
        hidden = self.transformer(hidden, attention_mask)
        
        memory_output = hidden[:, -self.memory_size:, :]
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        hidden = hidden[:, self.memory_size:-self.memory_size, :]
        if actions is not None:
            hidden = hidden.reshape(B, T, 3, self.hidden_dim).permute(0,2,1,3)
        else:
            hidden = hidden.reshape(B, T, 2, self.hidden_dim).permute(0,2,1,3)

        action_logits = self.action_predictor(hidden[:,1])

        return action_logits, memory_output

    # batch dimension always equal to 1 right now, will probably remove it in the future
    # Actions will be one timestep less, as we don't know the future action
    def inference(self, returns, states, actions, timesteps, attention_mask=None, past_memory=None):

        B, T = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timesteps)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # Embed states
        states_embeddings = self.state_embed(states.long())

        if actions is not None:
            actions = torch.cat([actions, torch.zeros(B, 1).to(actions.device).long()], dim=1)
            if actions.shape[1] > T:
                actions = actions[:,1:]
            actions_embeddings = torch.squeeze(self.embed_action(actions), 2) + time_embeddings

        if actions is not None:
            hidden = torch.stack([returns_embeddings, states_embeddings, actions_embeddings], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.hidden_dim)
        else:
            hidden = torch.stack([returns_embeddings, states_embeddings], dim=1).permute(0,2,1,3).reshape(B, 2*T, self.hidden_dim)
        info_T = hidden.shape[1]

        # Handle memory tokens
        if past_memory is None:
            hidden = torch.cat([self.read_mem_embedding.repeat(B, 1, 1), hidden, self.memory_tokens.repeat(B, 1, 1)], dim=1)
        else:
            read_tokens = past_memory + self.read_mem_embedding.repeat(B, 1, 1)
            hidden = torch.cat([read_tokens, hidden, past_memory], dim=1)

        hidden = self.embed_ln(hidden)
        if attention_mask is not None:
            ones = torch.ones((info_T, info_T))
            causal_mask = torch.tril(ones).view(1, info_T, info_T).cuda()
            attention_mask = causal_mask * attention_mask
            attention_mask = F.pad(attention_mask, (0, self.memory_size, self.memory_size, 0), value = False)
            attention_mask = F.pad(attention_mask, (self.memory_size, 0, 0, self.memory_size), value = True)
        else:
            ones = torch.ones((info_T, info_T))
            attention_mask = torch.tril(ones).view(1, info_T, info_T).cuda()
            attention_mask = F.pad(attention_mask, (0, self.memory_size, self.memory_size, 0), value = False)
            attention_mask = F.pad(attention_mask, (self.memory_size, 0, 0, self.memory_size), value = True)
        hidden = self.transformer(hidden, attention_mask)
        
        memory_output = hidden[:, -self.memory_size:, :]
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        hidden = hidden[:, self.memory_size:-self.memory_size, :]
        if actions is not None:
            hidden = hidden.reshape(B, T, 3, self.hidden_dim).permute(0,2,1,3)
        else:
            hidden = hidden.reshape(B, T, 2, self.hidden_dim).permute(0,2,1,3)

        action_logits = self.action_predictor(hidden[:,1])

        return action_logits, memory_output

    def convert_obs_inf(self, obs):
        states = torch.tensor(obs)
        states = states.unsqueeze(0).unsqueeze(0)

        return states