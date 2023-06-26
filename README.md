# Overview

This repo is based on [RL Starter Files](https://github.com/lcswillems/rl-starter-files) and [torch-ac](https://github.com/lcswillems/torch-ac)

<p align="center">
    <img width="300" src="README-rsrc/visualize-keycorridor.gif">
</p>

These files are suited for [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments and [`torch-ac`](https://github.com/lcswillems/torch-ac) RL algorithms. They are easy to adapt to other environments and RL algorithms.

## Installation

1. Clone this repository.

2. Install `gym-minigrid` environments and other necessary dependency

```
pip3 install -r requirements.txt
```


## Usage
In this repo, train/test are all done through config files.

```
# Train PPO
python3 -m scripts.train/evaluate/visualize --config path_to_config --custom_path if_custom_checkpoint_path
# Train Decision transformer
python3 -m decision_transformer.train/evaluate/visualize --config path_to_config --custom_path if_custom_checkpoint_path
```

You can also manually inspect the environment through GUI with
```
python manual_control.py --env MiniGrid-MemoryS13-v0-seeobj
```
