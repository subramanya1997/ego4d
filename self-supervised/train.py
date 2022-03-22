import os
from pickletools import optimize
from pyexpat import model
import sys
import json
import yaml
import argparse

import torch
import wandb

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam
from tqdm import tqdm

from model.model import *
from data_loader import MEMEDataLoader, Modal


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", default='config.yaml'
    )
    parser.add_argument("--max-len", help="maximum length of answer clip", type=int, default=None)
    parser.add_argument("--force-cpu", help="enforce cpu computation", type=int, default=None)
    parser.add_argument("-r", "--record-path", help="path for saving records", type=str, default='output/records/')
    parser.add_argument("-p", "--prefix", help="prefix for this run", type=str, default='meme')
    parser.add_argument("-l", "--loss-type", help="loss type to use", type=str, default='pos_loss')
    parser.add_argument("-w", "--wandb-name", \
                        help="wandb name for this run, defaults to random names by wandb",\
                        type=str, default=None)
    parser.add_argument("--model-save-path", help="path of the directory with model checkpoint", type=str, default=None)


    try:
        parsed_args = parser.parse_args()
    except (IOError) as msg:
        parser.error(str(msg))

    # Read config yamls file
    config_file = parsed_args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if key not in parsed_args.__dict__ or parsed_args.__dict__[key] is None:
            parsed_args.__dict__[key] = value

    if parsed_args.model_save_path is None:
        parsed_args.model_save_path = "./output/models/"
    parsed_args.best_model_path = os.path.join(parsed_args.model_save_path, f"{parsed_args.prefix}_{parsed_args.wandb_name}_best.pth")
    parsed_args.last_model_path = os.path.join(parsed_args.model_save_path, f"{parsed_args.prefix}_{parsed_args.wandb_name}_last.pth")

    if not parsed_args.force_cpu:
        parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Parsed Arguments are - ", parsed_args)

    return parsed_args

def initialise_wandb(args,model):
    wandb.init(project=f"ego4d_{args.prefix}", entity="ego4d-self", config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
    },resume = args.resume,settings=wandb.Settings(start_method="fork"))
    if args.wandb_name is not None:
        wandb.run.name = args.wandb_name
        wandb.run.save()

    wandb.watch(model)

def get_dataloader(args):
    print("Loading data")
    train = MEMEDataLoader(json_path=args.input_train_split, split="train")
    val = MEMEDataLoader(json_path=args.input_val_split, split="val")
    test = MEMEDataLoader(json_path=args.input_test_split, split="test")
    args = None
    train_loader = None 
    val_loader = None 
    test_loader = None
    return args, train_loader, val_loader, test_loader, train, val, test

def load_checkpoint(model, optimizer, path):
    wandb.restore(path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    mIoU = checkpoint['mIoU']
    print(f"Loaded checkpoint from {path}")
    return model, optimizer, epoch, loss, mIoU

def train():
    pass

if __name__ == "__main__":
    args = parse_arguments()
    args, train_loader, val_loader, test_loader, train_data, val_data, test_data = get_dataloader(args)

    # model, model_loss, optimizer = init_model(args)
    # initialise_wandb(args, model)
    # writer = SummaryWriter()

    # if wandb.run.resumed or args.resume:
    #     model, optimizer, start_epoch, loss, best_mIoU = load_checkpoint(model, optimizer, args.last_model_path)