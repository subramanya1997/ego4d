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
    parser.add_argument("-p", "--prefix", help="prefix for this run", type=str, default='meme_ss')
    parser.add_argument("-l", "--loss-type", help="loss type to use", type=str, default='Distance')
    parser.add_argument("-w", "--wandb-name", \
                        help="wandb name for this run, defaults to random names by wandb",\
                        type=str, default=None)
    parser.add_argument("--model-save-path", help="path of the directory with model checkpoint", type=str, default=None)
    parser.add_argument("--loss_weight", help="loss weight", type=float, default=0.25)
    parser.add_argument("--reduction", help="reduction", type=str, default='mean')

    
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



def get_dataloader(args):
    print("Loading data...")
    train = MEMEDataLoader(json_path=args.input_train_split, split="train", modalities=[Modal._Video, Modal._Audio], config_file=args.dataloader_config)
    val = MEMEDataLoader(json_path=args.input_val_split, split="val", modalities=[Modal._Video, Modal._Audio], config_file=args.dataloader_config)
    test = MEMEDataLoader(json_path=args.input_test_split, split="test", modalities=[Modal._Video, Modal._Audio], config_file=args.dataloader_config)

    train_loader = get_loader(train, batch_size=args.batch_size, type=args.dataloader_type)
    val_loader = get_loader(val, batch_size=args.batch_size, type=args.dataloader_type) 
    test_loader = get_loader(test, batch_size=args.batch_size, type=args.dataloader_type)

    args.video_feature_size = train.video_feature_size if train.video_feature_size is not None else 0
    args.query_feature_size = val.query_feature_size if val.query_feature_size is not None else 0
    args.audio_feature_size = test.audio_feature_size if test.audio_feature_size is not None else 0

    return args, train_loader, val_loader, test_loader, train, val, test

def init_model(args):
    print("Initializing model...")
    model = MEME_SS_BASE(args)
    model.to(args.device)

    model_loss = MEME_SS_LOSS(args)
    model_loss.to(args.device)

    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    return model, model_loss, optimizer

def get_modalities(args):
    modals = [Modal._Video,Modal._Transcript]
    if args.audio:
        modals.append(Modal._Audio)
    return modals

def accuracy(pred, gtruth, window):
    sub = pred - gtruth
    sub = torch.le(torch.abs(sub), window)
    _sum = torch.mean((sub == True).float())
    return _sum.item()

def train(model, dataloader, model_loss, optimizer, args, writer, epoch):
    model.train()
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="sample %3d / %3d" % (epoch + 1, args.num_epochs),
            )

    iter = 0
    total_loss = 0
    acc, acc5, acc10 = 0, 0, 0
    for data in tqdm_obj:
        _vid, video_features, audio_features, query_features, text_length, query_frame_numbers, query_data = data

        # indexs = torch.arange(0, video_features.shape[1])
        # if args.randomize:
        #     indexs = torch.randperm(video_features.shape[1])
        #     video_features[0] = video_features[0][indexs]
        #     gtruth = gtruth[indexs]

        video_features = video_features.to(args.device)
        audio_features = audio_features.to(args.device)
        query_frame_numbers = query_frame_numbers.to(args.device)
        
        output, frame_pred, reorder_pred, frame_number_pred, clsTokens = model(video_features, query_features, text_length, audio_features, args, modalities = get_modalities(args))
        
        loss = model_loss(frame_number_pred, query_frame_numbers, loss_type=args.loss_type)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.cpu().item()
        n_iter = epoch * len(dataloader) + iter

        acc += accuracy(frame_number_pred, query_frame_numbers, 1)
        acc5 += accuracy(frame_number_pred, query_frame_numbers, 5)
        acc10 += accuracy(frame_number_pred, query_frame_numbers, 10)
        
        
        iter += 1
        if iter%100 == 0:
            print(f'Loss/train/{epoch} : {loss.cpu().item()} ==> iter : {iter} ==> acc : {acc/iter} ==> acc 5 : {acc5/iter} ==> acc 10 : {acc10/iter}')
            # print(f'pred : {frame_number_pred[:5]} ==> gtruth : {query_frame_numbers[:5]}')
        
        writer.add_scalar('Loss/train', loss.detach().cpu().item(), n_iter)
        wandb.log({"Batch Loss": {"Train": loss.detach().cpu().item()}})
        wandb.log({"Batch accuracy": { "window 1/train": acc/iter, "window 5/train": acc5/iter, "window 10/train": acc10/iter}})

    acc = acc/iter
    acc5 = acc5/iter
    acc10 = acc10/iter

    print(f'Total Loss/train : {total_loss} ==> epoch : {epoch} ==> acc : {acc} ==> acc 5 : {acc5} ==> acc 10 : {acc10}')
    wandb.log({f"Total loss/train": total_loss})
    wandb.log({f"Accuracy": {"window 1/train": acc, "window 5/train": acc5, "window 10/train": acc10}})
    return total_loss, acc, acc5, acc10

def test_model(model, dataloader, model_loss, args, writer, epoch, Test = False):
    model.eval()
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Epoch %3d / %3d" % (epoch + 1, args.num_epochs),
            )
    iter = 0
    total_loss = 0
    acc, acc5, acc10 = 0, 0, 0
    for i, data in enumerate(tqdm_obj):
        _vid, video_features, audio_features, query_features, text_length, query_frame_numbers, query_data = data
        # indexs = torch.arange(0, video_features.shape[1])
        # if args.randomize:
        #     indexs = torch.randperm(video_features.shape[1])
        #     video_features[0] = video_features[0][indexs]
        #     gtruth = gtruth[indexs]

        video_features = video_features.to(args.device)
        audio_features = audio_features.to(args.device)
        query_frame_numbers = query_frame_numbers.to(args.device)

        with torch.no_grad():
            output, frame_pred, reorder_pred, frame_number_pred, clsTokens = model(video_features, query_features, text_length, audio_features, args, modalities = get_modalities(args))

        loss = model_loss(frame_number_pred, query_frame_numbers, loss_type=args.loss_type)
        
        total_loss += loss.cpu().item()

        acc += accuracy(frame_number_pred, query_frame_numbers, 1)
        acc5 += accuracy(frame_number_pred, query_frame_numbers, 5)
        acc10 += accuracy(frame_number_pred, query_frame_numbers, 10)

        n_iter = epoch * len(dataloader) + iter
        split = "test" if Test else "val"
        
        iter += 1

        if iter%100 == 0:
            print(f'Loss/{split}/{epoch} : {loss.cpu().item()} ==> iter : {iter} ==> acc : {acc/iter} ==> acc 5 : {acc5/iter} ==> acc 10 : {acc10/iter}')
        writer.add_scalar(f'Loss/{split}', loss.cpu().item(), n_iter)
        wandb.log({"Batch Loss": {f"{split}": loss.detach().cpu().item()}})
        wandb.log({"Batch accuracy": {f"window 1/{split}": acc/iter, f"window 5/{split}": acc5/iter, f"window 10/{split}": acc10/iter}})

    acc = acc/iter
    acc5 = acc5/iter
    acc10 = acc10/iter

    print(f'Total Loss/{split} : {total_loss} ==> epoch : {epoch}')
    wandb.log({f"Total loss/{split}": total_loss})
    wandb.log({f"Accuracy": {f"window 1/{split}": acc, f"window 5/{split}": acc5, f"window 10/{split}": acc10}})
    return total_loss, acc, acc5, acc10

def save_checkpoint(model, optimizer, epoch, loss, acc, acc5, acc10, path):
    torch.save({ # Save our checkpoint loc
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
            'acc5': acc5,
            'acc10': acc10,
            }, path)
    wandb.save(path)

def load_checkpoint(model, optimizer, path):
    wandb.restore(path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    acc5 = checkpoint['acc5']
    acc10 = checkpoint['acc10']
    print(f"Loaded checkpoint from {path}")
    return model, optimizer, epoch, loss, acc, acc5, acc10

def initialise_wandb(args, model):
    wandb.init(project="self-supervised", entity="ego4d-meme")
    wandb.config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
    }
    wandb.resume = args.resume
    wandb.settings = wandb.Settings(start_method="fork")

    if args.wandb_name is not None:
        wandb.run.name = args.wandb_name
        wandb.run.save()

    wandb.watch(model)

if __name__ == "__main__":
    args = parse_arguments()
    args, train_loader, val_loader, test_loader, train_data, val_data, test_data = get_dataloader(args)
    print("Done loading data...")
    args.embedding_dim = args.video_feature_size + args.query_feature_size 
    if args.audio:
        args.embedding_dim += args.audio_feature_size

    model, model_loss, optimizer = init_model(args)
    initialise_wandb(args, model)
    writer = SummaryWriter()

    best_val_loss = 100000000
    best_acc = 0

    if wandb.run.resumed or args.resume:
        model, optimizer, start_epoch, loss, best_accc, acc5, acc10 = load_checkpoint(model, optimizer, args.last_model_path)
        print(f"Loaded checkpoint from {args.last_model_path}")

    for epoch in range(0, args.num_epochs):
        train_loss, train_acc, train_acc5, train_acc10 = train(model, train_loader, model_loss, optimizer, args, writer, epoch)
        val_loss, val_acc, val_acc5, val_acc10 = test_model(model, val_loader, model_loss, args, writer, epoch, Test = False)
        test_loss, test_acc, test_acc5, test_acc10 = test_model(model, test_loader, model_loss, args, writer, epoch, Test = True)

        if best_acc < val_acc:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_acc5, val_acc10, args.best_model_path)
        
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_acc5, val_acc10, args.last_model_path)
    
