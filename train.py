import os
import tqdm
import json
import yaml
import argparse

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, ConstantLR

from model import MEME, MEME_LOSS
from utils import decode_candidate_clips

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", required=True
    )
    parser.add_argument(
        "--max-ans-len", help="maximum length of answer clip", type=int, default=None
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    
    # Read config yamls file
    config_file = parsed_args.pop("config_file")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if key not in parsed_args or parsed_args[key] is None:
            parsed_args[key] = value
    
    if not parsed_args.force_cpu:
        parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Parsed Arguments are - ", parse_args)

    return parsed_args

def get_dataloader(args):
    # train_loader = get_train_loader(
    #     dataset=dataset["train_set"], video_features=visual_features, configs=configs
    # )
    # train_eval_loader = get_test_loader(
    #     dataset=dataset["train_set"], video_features=visual_features, configs=configs
    # )
    # val_loader = (
    #     None
    #     if dataset["val_set"] is None
    #     else get_test_loader(dataset["val_set"], visual_features, configs)
    # )
    # test_loader = get_test_loader(
    #     dataset=dataset["test_set"], video_features=visual_features, configs=configs
    # )
    # configs.num_train_steps = len(train_loader) * configs.epochs
    # num_train_batches = len(train_loader)
    # num_val_batches = 0 if val_loader is None else len(val_loader)
    # num_test_batches = len(test_loader)
    train_loader = []
    val_loader = []
    test_loader = []
    return train_loader, val_loader, test_loader

def init_model(args):
    model = MEME(args)
    model.to(args.device)

    model_loss = MEME_LOSS(args)
    model_loss.to(args.device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    #scheduler = ConstantLR(optimizer, factor=0.95,total_iters=args.num_epochs*args.num_batches)
    
    return model, model_loss, optimizer

def infer_from_model(pred):
    start = pred[:, 0].cpu().numpy()
    end = pred[:, 1].cpu().numpy()
    max_len = args.max_len
    return decode_candidate_clips(start, end, args.topk, max_len)

def train(model, dataloader, model_loss, optimizer, args, writer, epoch):
    model.train()
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Epoch %3d / %3d" % (epoch + 1, args.num_epochs),
            )
    iter = 0
    total_loss = 0
    for data in tqdm_obj:
        (clip_id, features, starts, ends, query) = data
        features = features.to(args.device)
        starts = starts.to(args.device)
        ends = ends.to(args.device)
        query = query.to(args.device)

        pred = model(features)
        loss = model_loss(pred, starts, ends)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # Logging
        total_loss += loss.detach().cpu().item()
        n_iter = epoch * len(dataloader) + iter
        writer.add_scalar('Loss/train', loss.detach().cpu().item(), n_iter)
        iter += 1

        # end train loop

    return total_loss

def validate(model, dataloader, model_loss, args, writer, epoch):
    model.eval()
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Epoch %3d / %3d" % (epoch + 1, args.num_epochs),
            )
    iter = 0
    total_loss = 0
    for data in tqdm_obj:
        (clip_id, features, starts, ends, query) = data
        features = features.to(args.device)
        starts = starts.to(args.device)
        ends = ends.to(args.device)
        query = query.to(args.device)

        with torch.no_grad():
            pred = model(features)
            loss = model_loss(pred, starts, ends)

        # Logging
        total_loss += loss.cpu().item()
        n_iter = epoch * len(dataloader) + iter
        writer.add_scalar('Loss/val', loss.cpu().item(), n_iter)
        iter += 1

        # end val loop
    return total_loss

def test(model, dataloader, model_loss, args, writer):
    model.eval()
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Epoch %3d / %3d" % (epoch + 1, args.num_epochs),
            )
    iter = 0
    total_loss = 0
    for data in tqdm_obj:
        (clip_id, features, starts, ends, query) = data
        features = features.to(args.device)
        starts = starts.to(args.device)
        ends = ends.to(args.device)
        query = query.to(args.device)

        with torch.no_grad():
            pred = model(features)
            loss = model_loss(pred, starts, ends)

        # Logging
        total_loss += loss.cpu().item()
        n_iter = epoch * len(dataloader) + iter
        writer.add_scalar('Loss/test', loss.cpu().item(), n_iter)
        iter += 1

        # end test loop
    return total_loss

if __name__ == "__main__":
    args = parse_args()
    train_loader, val_loader, test_loader = get_dataloader(args)
    model, model_loss, optimizer = init_model(args)
    writer = SummaryWriter()

    for epoch in range(args.num_epochs):
        train_loss = train(model, model_loss, optimizer, args)
        val_loss = validate(model, model_loss, args)
        #evaluate
        #test if better results
        test_loss = test(model, model_loss, args)
