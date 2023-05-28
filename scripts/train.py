import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
import models
from configs.config import cfg, cfg_from_file

import envs

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

    # Load environments

    envs = []
    for i in range(cfg.procs):
        envs.append(utils.make_env(cfg.env_name, cfg.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    # obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    obs_space = envs[0].observation_space
    preprocess_obss = None
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = models.build_model_lstm(cfg, obs_space, envs[0].action_space)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if cfg.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, cfg.frames_per_proc, cfg.discount, cfg.lr, cfg.gae_lambda,
                                cfg.entropy_coef, cfg.value_loss_coef, cfg.max_grad_norm, cfg.recurrence,
                                cfg.optim_alpha, cfg.optim_eps, preprocess_obss)
    elif cfg.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, preprocess_obss, cfg)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(cfg.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    print(cfg.frames)
    while num_frames < cfg.frames:
        if num_frames < cfg.bc_period:
            bc_mode = True
        else:
            bc_mode = False
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences(bc_mode)
        # Manual lr scheduling
        # if (num_frames+1) % 20481 == 0:
        #     print("lr down")
        #     logs2 = algo.update_parameters(exps, bc_mode, lr_down=True)
        # else:
        logs2 = algo.update_parameters(exps, bc_mode)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % cfg.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if cfg.save_interval > 0 and update % cfg.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
