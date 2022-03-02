import os
import tqdm
import json
import yaml
import argparse

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from model.meme import MEME
from model.meme_loss import MEME_LOSS
from utils.metrics import decode_candidate_clips
from utils.data_processing import Ego4d_NLQ, get_train_loader, get_test_loader

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", required=True
    )
    parser.add_argument(
        "--max-ans-len", help="maximum length of answer clip", type=int, default=None
    )
    parser.add_argument(
        "--force-cpu", help="enforce cpu computation", type=int, default=None
    )

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
            parsed_args.key = value
    
    if not parsed_args.force_cpu:
        parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Parsed Arguments are - ", parse_args)

    return parsed_args

def get_dataloader(args):
    train_nql = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_train.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="train", wordEmbedding="bert", number_of_sample=1000, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/train.pkl")
    val_nql = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_val.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="val", wordEmbedding="bert", number_of_sample=1000, save_or_load=False, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/val.pkl")

    train_loader = get_train_loader(train_nql, batch_size=1)
    val_loader = get_test_loader(val_nql, batch_size=1)
    test_loader = []

    video_feature_size, query_feature_size = train_nql.video_feature_size, train_nql.query_feature_size

    return train_loader, val_loader, test_loader, video_feature_size, query_feature_size

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
        (clip_id, features, query_emb, starts, ends, query) = data
        features = features.to(args.device)
        query_emb = query_emb.to(args.device)
        starts = starts.to(args.device)
        ends = ends.to(args.device)
        query = query.to(args.device)

        pred = model(features, query_emb)
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
    train_loader, val_loader, test_loader, video_feature_size, query_feature_size = get_dataloader(args)
    args.embedding_dim = video_feature_size + query_feature_size
    model, model_loss, optimizer = init_model(args)
    writer = SummaryWriter()

    for epoch in range(args.num_epochs):
        train_loss = train(model, model_loss, optimizer, args)
        val_loss = validate(model, model_loss, args)
        #evaluate
        #test if better results
        test_loss = test(model, model_loss, args)
