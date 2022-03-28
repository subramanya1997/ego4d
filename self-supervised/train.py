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

from model.model import MEME_SS_BASE
from model.loss import MEME_SS_LOSS
from model.data_loader import MEMEDataLoader, Modal, get_loader


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
    }, resume = args.resume, settings = wandb.Settings(start_method="fork"))
    if args.wandb_name is not None:
        wandb.run.name = args.wandb_name
        wandb.run.save()

    wandb.watch(model)

def get_dataloader(args):
    print("Loading data...")
    train = MEMEDataLoader(json_path=args.input_train_split, split="train", modalities=[Modal._Video, Modal._Audio], config_file=args.dataloader_config)
    val = MEMEDataLoader(json_path=args.input_val_split, split="val", modalities=[Modal._Video, Modal._Audio], config_file=args.dataloader_config)
    test = MEMEDataLoader(json_path=args.input_test_split, split="test", modalities=[Modal._Video, Modal._Audio], config_file=args.dataloader_config)

    train_loader = get_loader(train, batch_size=5)
    val_loader = get_loader(val, batch_size=5) 
    test_loader = get_loader(test, batch_size=5)

    args.video_feature_size = train.video_feature_size if train.video_feature_size is not None else 0
    args.query_feature_size = val.query_feature_size if val.query_feature_size is not None else 0
    args.audio_feature_size = test.audio_feature_size if test.audio_feature_size is not None else 0

    return args, train_loader, val_loader, test_loader, train, val, test

def init_model(args):
    print("Initializing model...")
    model = MEME_SS_BASE(args)
    model.to(args.device)

    model_loss = MEME_SS_LOSS(args)
    #model_loss.to(args.device)

    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    return model, model_loss, optimizer

def load_checkpoint(model, optimizer, path):
    wandb.restore(path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {path}")
    return model, optimizer, epoch, loss

def get_modalities(args):
    modals = [Modal._Video,Modal._Transcript]
    if args.audio:
        modals.append(Modal._Audio)
    return modals

def train(model, dataloader, model_loss, optimizer, args, writer, epoch):
    model.train()
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Epoch %3d / %3d" % (epoch + 1, args.num_epochs),
            )

    iter = 0
    total_loss = 0
    issue_cids = []
    for data in tqdm_obj:
        #print(data)
        _vid, video_features, audio_features, query_features, info = data
        video_features = video_features.to(args.device)
        audio_features = audio_features.to(args.device)
        query_features = query_features.to(args.device)
        pred, framePred = model(video_features, query_features, audio_features, modalities = get_modalities(args))
        print(framePred.shape, info)
        pass

if __name__ == "__main__":
    args = parse_arguments()
    args, train_loader, val_loader, test_loader, train_data, val_data, test_data = get_dataloader(args)
    print("Done loading data...")
    args.embedding_dim = args.video_feature_size + args.query_feature_size 
    if args.audio:
        args.embedding_dim += args.audio_feature_size

    model, model_loss, optimizer = init_model(args)
    #initialise_wandb(args, model)
    writer = SummaryWriter()

    # if wandb.run.resumed or args.resume:
    #     model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, args.last_model_path)

    for epoch in range(0, args.num_epochs):
        train_loss = train(model, train_loader, model_loss, optimizer, args, writer, epoch)

    