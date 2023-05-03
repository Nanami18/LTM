import torch

def default_preprocess_obss(obss, multi_frame=False, device=None):
    if multi_frame:
        return torch.stack([torch.tensor(obs) for obs in obss])
    return torch.tensor(obss, device=device)