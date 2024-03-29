import argparse
import time
import datetime
import torch_ac
import torch
from torch.utils.data import DataLoader
import numpy as np
import tensorboardX
import sys

import utils
from utils import device
from configs.config import cfg, cfg_from_file

import envs
from models.decision_transformer_model import build_model
from decision_transformer.dataset_generation import (
    generate_expert_trajectories, 
    generate_random_trajectories, 
    HallwayMemoryEnvDataset, 
    collate_fn_findingobj,
    FindingObjEnvDataset,
)

def basic_dt_training_loop(cfg, model, train_loader, optimizer, txt_logger, model_dir, status):
    model.train()
    trained_frames = status["num_frames"]
    epoch_count = status['update']
    start_time = time.time()
    while trained_frames < cfg.frames:
        cum_loss = 0
        for batch in train_loader:
            states, actions, rewards, timesteps, masks = batch
            states = {k:v.to(device) for k, v in states.items()}
            actions = actions.to(device)
            gt_actions = actions.detach().clone()
            rewards = rewards.to(device)
            timesteps = timesteps.to(device)
            masks = masks.to(device)

            action_logits = model(rewards, states, actions, timesteps, masks)
            # Compute loss and update the model
            if cfg.discrete_action:
                # Ignore padding when calculating the loss
                gt_actions = torch.where(masks[:,0,0,::3].unsqueeze(2) == 1, gt_actions, torch.tensor(-1).to(device))
                # Downweight forward actions
                if "Memory" in cfg.env_name:
                    weight = torch.tensor([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]).cuda()
                    loss = torch.nn.functional.cross_entropy(action_logits.permute(0, 2, 1), gt_actions.squeeze(2), ignore_index=-1, weight=weight)
                elif "ObjLocate" in cfg.env_name:
                    weight = torch.tensor([1.0, 1.0, 0.1, 0.1, 5.0, 0.1, 0.1]).cuda()
                    loss = torch.nn.functional.cross_entropy(action_logits.permute(0, 2, 1), gt_actions.squeeze(2), ignore_index=-1)
            else:
                loss = torch.nn.functional.mse_loss(action_logits, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            cum_loss += loss.item()

            trained_frames += states['target'].shape[0]
            
        epoch_count += 1
        if epoch_count % cfg.log_interval == 0:
            training_time = time.time() - start_time
            txt_logger.info(f"Epoch {epoch_count} | Loss: {cum_loss} | Training time: {training_time}")

        if epoch_count % cfg.save_interval == 0:
            status = {"num_frames": trained_frames, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "update": epoch_count}
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

    status = {"num_frames": trained_frames, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "update": epoch_count}
    return status


def rmt_dt_training_loop(cfg, model, train_loader, optimizer, txt_logger, model_dir, status):
    model.train()
    trained_frames = status["num_frames"]
    epoch_count = status['update']
    start_time = time.time()
    while trained_frames < cfg.frames:
        cum_loss = 0
        for batch in train_loader:
            states, actions, rewards, timesteps, masks = batch
            states = states.to(device)
            actions = actions.to(device)
            gt_actions = actions.detach().clone()
            rewards = rewards.to(device)
            timesteps = timesteps.to(device)
            masks = masks.to(device)
            action_predictions = torch.zeros((states.shape[0], states.shape[1], env.action_space.n)).to(device)

            time_break = range(0, states.shape[1], cfg.context_length)
            memory = None
            for i in range(len(time_break)):
                cur_states = states[:, cfg.context_length*i:cfg.context_length*(i+1)]
                cur_actions = actions[:, cfg.context_length*i:cfg.context_length*(i+1)]
                cur_rewards = rewards[:, cfg.context_length*i:cfg.context_length*(i+1)]
                cur_timesteps = timesteps[:, cfg.context_length*i:cfg.context_length*(i+1)]
                cur_masks = masks[:, :, cfg.context_length*i*3:cfg.context_length*(i+1)*3, cfg.context_length*i*3:cfg.context_length*(i+1)*3]
                action_logits, memory = model(cur_rewards, cur_states, cur_actions, cur_timesteps, cur_masks, memory)
                action_predictions[:, cfg.context_length*i:cfg.context_length*(i+1)] = action_logits

            # Compute loss and update the model
            # print(gt_actions[0,-2,0])
            # print(action_predictions[0,-2,:])
            if cfg.discrete_action:
                # Ignore padding when calculating the loss
                gt_actions = torch.where(masks[:,0,0,::3].unsqueeze(2) == 1, gt_actions, torch.tensor(-1).to(device))
                # Downweight forward actions
                weight = torch.tensor([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]).cuda()
                loss = torch.nn.functional.cross_entropy(action_predictions.permute(0, 2, 1), gt_actions.squeeze(2), ignore_index=-1, weight=weight)
            else:
                loss = torch.nn.functional.mse_loss(action_predictions, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            cum_loss += loss.item()

            trained_frames += states.shape[0]
            
        epoch_count += 1
        if epoch_count % cfg.log_interval == 0:
            training_time = time.time() - start_time
            txt_logger.info(f"Epoch {epoch_count} | Loss: {cum_loss} | Training time: {training_time}")

        if epoch_count % cfg.save_interval == 0:
            status = {"num_frames": trained_frames, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "update": epoch_count}
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

    status = {"num_frames": trained_frames, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "update": epoch_count}
    return status
    

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--custom_dir", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    cfg_from_file(args.config)
    
    cfg.mem = cfg.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{cfg.env_name}_{cfg.algo}_seed{cfg.seed}_{date}"
    
    if args.custom_dir is not None:
        model_name = args.custom_dir
    else:
        model_name = str(args.config).split("/")[-1][:-5] or default_model_name
    model_dir = utils.get_model_dir(model_name, cfg.env_name)
    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(cfg))

    # Set seed for all randomness sources

    utils.seed(cfg.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Create env
    env = utils.make_env(cfg.env_name, cfg.seed)

    # Load model
    model = build_model(cfg, env.observation_space, env.action_space)
    model.to(device)
    if "model_state" in status:
        model.load_state_dict(status["model_state"])
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(model))

    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.optim_eps)
    if "optimizer_state" in status:
        optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Load expert dataset
    trajectories = generate_expert_trajectories(env, cfg.num_episodes, cfg.reward_pertubation)
    print("Estimated epochs: {}".format(cfg.frames/cfg.num_episodes))
    if "Memory" in cfg.env_name:
        train_ds = HallwayMemoryEnvDataset(trajectories, cfg)
    elif "ObjLocate" in cfg.env_name:
        train_ds = FindingObjEnvDataset(trajectories, cfg)

    # Build dataloader
    if "ObjLocate" in cfg.env_name:
        collate_fn = collate_fn_findingobj
    else:
        collate_fn = None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    if cfg.use_rmt:
        status = rmt_dt_training_loop(cfg, model, train_loader, optimizer, txt_logger, model_dir, status)
    else:
        status = basic_dt_training_loop(cfg, model, train_loader, optimizer, txt_logger, model_dir, status)
    utils.save_status(status, model_dir)
    txt_logger.info("Training Completed, saved final checkpoint")