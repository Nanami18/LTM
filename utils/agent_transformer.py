import torch

import utils
from .other import device
from transformer_model import TransformerModel


class TransformerAgent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, cfg, num_envs=1):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = TransformerModel(obs_space, action_space, num_decoder_layers=cfg.Model.num_decoder_layers, 
            n_head=cfg.Model.n_head, recurrence=cfg.Training.recurrence)
        self.argmax = cfg.Inference.argmax
        self.num_envs = num_envs

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))
        
        # Have to hardcode the context length since currently the training framework is not designed to feed this here
        self.memories = torch.zeros(1, 8, self.acmodel.image_embedding_size, device=device)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            dist, _, memories = self.acmodel(preprocessed_obss, self.memories, torch.full((1,), 7))
            self.memories = torch.cat((self.memories[:, 1:, ...], memories), dim=1)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        # print(actions)
        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
