import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    DIR_TO_VEC
)

import functools 

def build_model(cfg, obs_space, action_space):
    if "ObjLocate" in cfg.env_name:
        if cfg.use_linear_procs:
            model = ACModelWithLinearProcs_findobj(obs_space, action_space, cfg)
        else:
            model = ACModelWithEmbed_findobj(obs_space, action_space, cfg)
    elif "Memory" in cfg.env_name:
        if "scalarobs" in cfg.env_name:
            model = ACModelWithScalarObs(obs_space, action_space, cfg)
        else:
            if cfg.use_embed:
                if cfg.use_linear_procs:
                    model = ACModelWithLinearProcs(obs_space, action_space, cfg)
                else:
                    model = ACModelWithEmbed(obs_space, action_space,cfg)
    else:
        model = ACModel(obs_space, action_space, cfg)
    
    return model

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m, cfg):
    classname = m.__class__.__name__
    if (classname.find("Linear") != -1 and classname.find("ACModel") == -1) or classname.find("Embedding") != -1 :
        m.weight.data.normal_(0, 1)
        if cfg.init_var:
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if classname != "Embedding" and m.bias is not None:
            m.bias.data.fill_(0)

class ACModelWithLinearProcs(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__()

        # Decide which components are enabled
        self.use_text = cfg.use_text
        self.use_memory = cfg.use_memory
       
        self.object_embed = nn.Embedding(len(OBJECT_TO_IDX), cfg.token_embed_size)
        self.image_procs = nn.Sequential(nn.Linear(cfg.token_embed_size, cfg.image_embed_size), nn.ReLU(), 
                nn.Linear(cfg.image_embed_size, cfg.image_embed_size))

        self.image_embedding_size = cfg.image_embed_size

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params, cfg)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        
        x = obs.image
        x = torch.flatten(x[:, :, :, 0], start_dim=1)
        x = torch.where(torch.sum(x == 5, dim=1) > torch.sum(x == 6, dim=1), torch.tensor(1).cuda(), 
            torch.where(torch.sum(x == 5, dim=1) < torch.sum(x == 6, dim=1), torch.tensor(2).cuda(), torch.tensor(0).cuda()))
        im_embed = self.object_embed(x)
        x = self.image_procs(im_embed)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class ACModelWithEmbed(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__()

        # Decide which components are enabled
        self.use_text = cfg.use_text
        self.use_memory = cfg.use_memory
       
        # Define image embedding
        self.object_embed = nn.Embedding(len(OBJECT_TO_IDX), cfg.token_embed_size)
        self.color_embed = nn.Embedding(len(COLOR_TO_IDX), cfg.token_embed_size)
        self.state_embed = nn.Embedding(2, cfg.token_embed_size)

        self.image_conv = nn.Sequential(
            nn.Conv2d(cfg.token_embed_size*3, cfg.image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(cfg.image_embed_size//2, cfg.image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(cfg.image_embed_size//2, cfg.image_embed_size, (2, 2)),
            nn.ReLU()
        )

        self.image_embedding_size = cfg.image_embed_size

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1)
        )

        # Initialize parameters correctly
        init_params(self, cfg)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        
        x = obs.image
        object_embedding = self.object_embed(x[:, :, :, 0].long())
        color_embedding = self.color_embed(x[:, :, :, 1].long())
        state_embedding = self.state_embed(x[:, :, :, 2].long())
        im_embed = torch.cat((color_embedding, object_embedding, state_embedding), dim=3)
        im_embed = im_embed.transpose(1,3).transpose(2,3)
        x = self.image_conv(im_embed)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class ACModelWithScalarObs(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__()

        # Decide which components are enabled
        self.use_text = cfg.use_text
        self.use_memory = cfg.use_memory
        self.hidden_dim = cfg.hidden_dim
       
        self.state_embed = nn.Embedding(obs_space.n, cfg.hidden_dim)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(cfg.hidden_dim, self.semi_memory_size)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 1)
        )

        # Initialize parameters correctly
        init_params(self, cfg)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.hidden_dim
    
    def forward(self, obs, memory):
        
        x = self.state_embed(obs)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class ACModelWithEmbed_findobj(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__()

        # Decide which components are enabled
        self.use_text = cfg.use_text
        self.use_memory = cfg.use_memory
       
        # Define image embedding
        self.object_embed = nn.Embedding(len(OBJECT_TO_IDX), cfg.token_embed_size)
        self.color_embed = nn.Embedding(len(COLOR_TO_IDX), cfg.token_embed_size)

        self.image_conv = nn.Sequential(
            nn.Conv2d(cfg.token_embed_size*2, cfg.image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(cfg.image_embed_size//2, cfg.image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(cfg.image_embed_size//2, cfg.image_embed_size, (2, 2)),
            nn.ReLU()
        )

        self.image_embedding_size = cfg.image_embed_size
        self.token_embed_size = cfg.token_embed_size

        # Define memory
        if self.use_memory:
            if cfg.num_lstm_layers == 0:
                self.layer_multiplier = 1
                self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            else:
                self.layer_multiplier = cfg.num_lstm_layers
                self.memory_rnn = nn.LSTM(self.image_embedding_size, self.image_embedding_size, cfg.num_lstm_layers, batch_first=True, dropout=cfg.lstm_dropout)
            self.num_lstm_layers = cfg.num_lstm_layers

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size + self.token_embed_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1)
        )

        # Initialize parameters correctly
        init_params(self, cfg)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size * self.layer_multiplier

    def forward(self, obs, memory):
        
        x = obs.image
        target_color = obs.target_color

        object_embedding = self.object_embed(x[:, :, :, 0].long())
        color_embedding = self.color_embed(x[:, :, :, 1].long())

        im_embed = torch.cat((color_embedding, object_embedding), dim=3)
        im_embed = im_embed.transpose(1,3).transpose(2,3)
        x = self.image_conv(im_embed)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            if self.num_lstm_layers == 0:
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                h = memory[:, :self.semi_memory_size].reshape(memory.shape[0], self.layer_multiplier, self.image_embedding_size).transpose(0,1).contiguous()
                c = memory[:, self.semi_memory_size:].reshape(memory.shape[0], self.layer_multiplier, self.image_embedding_size).transpose(0,1).contiguous()
                hidden = (h, c)
                _, hidden = self.memory_rnn(x.unsqueeze(1), hidden)
                embedding = hidden[0].transpose(0,1).reshape(memory.shape[0], -1)
                memory = torch.cat(hidden, dim=0).transpose(0,1).reshape(memory.shape[0], -1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)
        
        target_embedding = self.color_embed(target_color.long())
        embedding = torch.cat((embedding, target_embedding), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class ACModelWithLinearProcs_findobj(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__()

        # Decide which components are enabled
        self.use_text = cfg.use_text
        self.use_memory = cfg.use_memory
       
        # Target embedding
        self.target_embed = nn.Embedding(len(COLOR_TO_IDX), cfg.token_embed_size)

        # State embedding
        if isinstance(obs_space['image'], tuple):
            self.state_embed = nn.Sequential(
                nn.Linear(functools.reduce(lambda x, y : x*y, obs_space['image']), cfg.image_embed_size),
                nn.ReLU(),
                nn.Linear(cfg.image_embed_size, cfg.image_embed_size),
            )
        else:
            self.state_embed = nn.Sequential(
                nn.Linear(functools.reduce(lambda x, y : x*y, obs_space['image'].shape), cfg.image_embed_size),
                nn.ReLU(),
                nn.Linear(cfg.image_embed_size, cfg.image_embed_size),
            )

        self.image_embedding_size = cfg.image_embed_size
        self.token_embed_size = cfg.token_embed_size

        # Define memory
        if self.use_memory:
            if cfg.num_lstm_layers == 0:
                self.layer_multiplier = 1
                self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            else:
                self.layer_multiplier = cfg.num_lstm_layers
                self.memory_rnn = nn.LSTM(self.image_embedding_size, self.image_embedding_size, cfg.num_lstm_layers, batch_first=True, dropout=cfg.lstm_dropout)
            self.num_lstm_layers = cfg.num_lstm_layers

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size + self.token_embed_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1)
        )

        # Initialize parameters correctly
        init_params(self, cfg)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size * self.layer_multiplier

    def forward(self, obs, memory):
        
        x = obs.image
        target_color = obs.target_color
        x = self.state_embed(torch.flatten(x, start_dim=1))

        if self.use_memory:
            if self.num_lstm_layers == 0:
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                h = memory[:, :self.semi_memory_size].reshape(memory.shape[0], self.layer_multiplier, self.image_embedding_size).transpose(0,1).contiguous()
                c = memory[:, self.semi_memory_size:].reshape(memory.shape[0], self.layer_multiplier, self.image_embedding_size).transpose(0,1).contiguous()
                hidden = (h, c)
                _, hidden = self.memory_rnn(x.unsqueeze(1), hidden)
                embedding = hidden[0].transpose(0,1).reshape(memory.shape[0], -1)
                memory = torch.cat(hidden, dim=0).transpose(0,1).reshape(memory.shape[0], -1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)
        
        target_embedding = self.target_embed(target_color.long())
        embedding = torch.cat((embedding, target_embedding), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__()

        # Decide which components are enabled
        self.use_text = cfg.use_text
        self.use_memory = cfg.use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params, cfg)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]