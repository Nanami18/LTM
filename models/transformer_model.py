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

def build_model(cfg, obs_space, action_space):
    if cfg.use_pastkv:
        model = TransformerModel_UsePastKV(obs_space, action_space,
            cfg.token_embed_size, cfg.image_embed_size, cfg.num_decoder_layers, cfg.n_head, cfg.recurrence)
    else:
        model = TransformerModel(obs_space, action_space,
            cfg.token_embed_size, cfg.image_embed_size, cfg.num_decoder_layers, cfg.n_head, cfg.recurrence)
    
    return model

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class TransformerModel_UsePastKV(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, embed_size=16, image_embed_size=128, num_decoder_layers=2, n_head=4, recurrence=8):
        super().__init__()
       
        # Define image embedding
        self.object_embed = nn.Embedding(len(OBJECT_TO_IDX), embed_size)
        self.color_embed = nn.Embedding(len(COLOR_TO_IDX), embed_size)
        self.state_embed = nn.Embedding(2, embed_size)

        self.image_conv = nn.Sequential(
            nn.Conv2d(embed_size*3, image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(image_embed_size//2, image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(image_embed_size//2, image_embed_size, (2, 2)),
            nn.ReLU()
        )

        self.image_embedding_size = image_embed_size

        # Define memory
        self.num_decoder_layers = num_decoder_layers
        self.recurrence = recurrence
        self.decoder = nn.ModuleList()
        self.pos_embed = nn.Embedding(recurrence, image_embed_size)
        [self.decoder.append(nn.TransformerDecoderLayer(image_embed_size, n_head, batch_first=True)) for _ in range(num_decoder_layers)]


        # Resize image embedding
        self.embedding_size = image_embed_size

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
        self.apply(init_params)

    @property
    def memory_size(self):
        return self.image_embedding_size

    @property
    def observation_encoding_size(self):
        return self.image_embedding_size
    
    # memory is supposed to be a list of memory corresponds to each layer
    def forward(self, obs, memory):
        
        x = obs.image
        object_embedding = self.object_embed(x[:, :, :, 0].long())
        color_embedding = self.color_embed(x[:, :, :, 1].long())
        state_embedding = self.state_embed(x[:, :, :, 2].long())
        im_embed = torch.cat((color_embedding, object_embedding, state_embedding), dim=3)
        im_embed = im_embed.transpose(1,3).transpose(2,3)
        per_layer_input = []
        x = self.image_conv(im_embed)
        # Introduce sequence dimension
        x = x.reshape(x.shape[0], -1).unsqueeze(1)
        x = x + self.pos_embed(torch.arange(x.shape[1]).to(x.device))
        per_layer_input.append(x)
        
        for i in range(self.num_decoder_layers):
            x = self.decoder[i](x, memory[:, :, i, :])
            if i != self.num_decoder_layers-1:
                per_layer_input.append(x)

        cur_memory = torch.stack(per_layer_input, dim=2)
        memory = torch.cat((memory, cur_memory), dim=1)
        action = self.actor(x).squeeze(1)
        dist = Categorical(logits=F.log_softmax(action, dim=1))

        value = self.critic(x)
        value = value.squeeze(1).squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class TransformerModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, embed_size=16, image_embed_size=128, num_decoder_layers=2, n_head=4, recurrence=8):
        super().__init__()
       
        # Define image embedding
        self.object_embed = nn.Embedding(len(OBJECT_TO_IDX), embed_size)
        self.color_embed = nn.Embedding(len(COLOR_TO_IDX), embed_size)
        self.state_embed = nn.Embedding(2, embed_size)

        self.image_conv = nn.Sequential(
            nn.Conv2d(embed_size*3, image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(image_embed_size//2, image_embed_size//2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(image_embed_size//2, image_embed_size, (2, 2)),
            nn.ReLU()
        )

        self.image_embedding_size = image_embed_size

        # Define memory
        self.num_decoder_layers = num_decoder_layers
        self.n_head = n_head
        self.recurrence = recurrence
        self.decoder = nn.ModuleList()
        self.pos_embed = nn.Embedding(recurrence, image_embed_size)
        [self.decoder.append(nn.TransformerEncoderLayer(image_embed_size, n_head, batch_first=True)) for _ in range(num_decoder_layers)]


        # Resize image embedding
        self.embedding_size = image_embed_size

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
        self.apply(init_params)

    @property
    def memory_size(self):
        return self.image_embedding_size

    @property
    def observation_encoding_size(self):
        return self.image_embedding_size
    
    # memory is supposed to be a list of memory corresponds to each layer
    def forward(self, obs, memory, act_ind):
        
        x = obs.image
        object_embedding = self.object_embed(x[:, :, :, 0].long())
        color_embedding = self.color_embed(x[:, :, :, 1].long())
        state_embedding = self.state_embed(x[:, :, :, 2].long())
        im_embed = torch.cat((color_embedding, object_embedding, state_embedding), dim=3)
        im_embed = im_embed.transpose(1,3).transpose(2,3)
        x = self.image_conv(im_embed)
        # Introduce sequence dimension
        x = x.reshape(x.shape[0], -1).unsqueeze(1)
        cur_memory = x.clone()
        if memory is not None:
            proc_mem = []
            for i in range(memory.shape[0]):
                if act_ind[i] == 0:
                    # Initialize padding memory and put current observation at the beginning
                    cur_mem = torch.zeros(self.recurrence, memory.shape[2]).to(memory.device)
                    cur_mem[0] = x[i]
                    proc_mem.append(cur_mem)
                else:
                    # Put the current observation at the right index, put memory to the left and add padding to the right
                    padding = torch.zeros(self.recurrence-act_ind[i], memory.shape[2], device=memory.device)
                    cur_mem = torch.cat((memory[i][-act_ind[i]:], padding), dim=0)
                    cur_mem[act_ind] = x[i]
                    proc_mem.append(cur_mem)
            x = torch.stack(proc_mem)
        x = x + self.pos_embed(torch.arange(x.shape[1]).to(x.device))
        
        # causal attention mask
        attn_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
        # print("trans encoder input shape: ", x.shape)
        for i in range(self.num_decoder_layers):
            x = self.decoder[i](x, attn_mask)

        x = x[range(act_ind.shape[0]), act_ind, :]
        action = self.actor(x)
        dist = Categorical(logits=F.log_softmax(action, dim=1))

        value = self.critic(x)
        value = value.squeeze(1)
        return dist, value, cur_memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]