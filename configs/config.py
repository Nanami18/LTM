import os
import os.path as osp
import numpy as np
import math
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Environment
__C.env_name = "MiniGrid-MemoryS9-v0-seeobj"

# Training
__C.bc_period = 0
__C.reshape_reward = False
__C.algo = "ppo"
__C.seed = 1
__C.log_interval = 1
__C.save_interval = 10
__C.procs = 16
__C.frames = 1000000
__C.lr = 0.001
__C.gae_lambda = 0.95
__C.epochs = 4
__C.batch_size = 256
__C.frames_per_proc = 128
__C.discount = 0.99
__C.entropy_coef = 0.01
__C.value_loss_coef = 0.5
__C.max_grad_norm = 0.5
__C.optim_eps = 0.00000001
__C.optim_alpha = 0.99
__C.clip_eps = 0.2
__C.recurrence = 8
__C.text = False
__C.teacher_forcing = True

# Model
__C.use_memory = True
__C.cheat = False
__C.use_lstm = False
__C.use_linear_procs = False
__C.use_ext_mem = False
__C.token_embed_size = 16
__C.image_embed_size = 128
__C.use_text = False

# LSTM
__C.use_embed = True

# Transfomrer
__C.num_decoder_layers = 1
__C.n_head = 4
__C.use_pastkv = False

# Inference
__C.shift = 0
__C.argmax = False
__C.pause = 0.1
__C.gif = False
__C.episodes = 1000000
__C.worst_episodes_to_show = 10


def get_exp_dir(cfg_name):
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, cfg_name))
    return path

def get_output_dir(cfg_name, image_set, net_type):
    """Return the directory where experimental artifacts are placed.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    exp_dir = __C.REF_EXP_DIR if net_type == 'ref' else __C.DET_EXP_DIR
    config_output_path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', exp_dir, cfg_name))
    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path, '_'.join(image_sets))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    return final_output_path

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return
    
    if not set(a.keys()) == (b.keys()):
        print(f"Extra key from config file: {set(a.keys()).difference(set(b.keys()))}")
        print(f"Missing key from config file: {set(b.keys()).difference(set(a.keys()))}")

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)


def write_selected_class_file(filename, index):
    # read file
    with open(filename) as f:
        lines = [x for x in f.readlines()]
    lines_selected = [lines[i] for i in index]

    # write new file
    filename_new = filename + '.selected'
    f = open(filename_new, 'w')
    for i in range(len(lines_selected)):
        f.write(lines_selected[i])
    f.close()
    return filename_new
