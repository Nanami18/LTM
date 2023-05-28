import csv
import os
import torch
import logging
import sys

import utils
from .other import device


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir(env_name):
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    if "Memory" in env_name:
        if "scalarobs" in env_name:
            return "hallway_scalar_storage"
        else:
            return "storage"
    elif "ObjLocate" in env_name:
        return "find_storage"


def get_model_dir(model_name, env_name):
    return os.path.join(get_storage_dir(env_name), model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    txt_logger = logging.getLogger()
    txt_logger.setLevel(logging.INFO)

    # formatter = logging.Formatter('%(message)s')
    # file_handler = logging.FileHandler(filename=path)
    # file_handler.setFormatter(formatter)
    # txt_logger.addHandler(file_handler)
    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setFormatter(formatter)
    # txt_logger.addHandler(stream_handler)

    return txt_logger


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
