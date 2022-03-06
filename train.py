import os
import json
import yaml
import argparse

import torch
import wandb

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam
from tqdm import tqdm
from transformers import pipeline

from model.meme import MEME
from model.meme_loss import MEME_LOSS
from utils.metrics import decode_candidate_clips
from utils.data_processing import Ego4d_NLQ, get_train_loader, get_test_loader

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", default='config.yaml'
    )
    parser.add_argument("--max-len", help="maximum length of answer clip", type=int, default=None)
    parser.add_argument("--force-cpu", help="enforce cpu computation", type=int, default=None)

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
    
    if not parsed_args.force_cpu:
        parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Parsed Arguments are - ", parsed_args)

    return parsed_args

def get_dataloader(args):
    print("Loading data")
    train_nlq = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_train.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="train", wordEmbedding="bert", number_of_sample=1000, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/train_50.pkl")
    val_nlq = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_val.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="val", wordEmbedding="bert", number_of_sample=1000, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/val_25.pkl")
    test_nlq = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_val.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="test", wordEmbedding="bert", number_of_sample=1000, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/test_25.pkl")

    train_loader = get_train_loader(train_nlq, batch_size=1)
    val_loader = get_test_loader(val_nlq, batch_size=1)
    test_loader = get_test_loader(test_nlq, batch_size=1)

    video_feature_size, query_feature_size = 2304, 768# train_nlq.video_feature_size, train_nlq.query_feature_size TODO
    print("Finished loading data")
    return train_loader, val_loader, test_loader, video_feature_size, query_feature_size

def init_model(args):
    print("Initializing model")
    model = MEME(args)
    model.to(args.device)

    model_loss = MEME_LOSS(args)
    model_loss.to(args.device)

    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    #scheduler = ConstantLR(optimizer, factor=0.95,total_iters=args.num_epochs*args.num_batches)
    print("Finished initializing model")

    return model, model_loss, optimizer

def infer_from_model(pred, topk, qa_pipeline):
    start = pred[:, 0].cpu().numpy()
    end = pred[:, 1].cpu().numpy()
    max_len = args.max_len
    s, e, scores = decode_candidate_clips(qa_pipeline, start, end, topk, max_len)
    return s, e, scores

def train(model, dataloader, model_loss, optimizer, args, writer, epoch):
    model.train()
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Epoch %3d / %3d" % (epoch + 1, args.num_epochs),
            )
    iter = 0
    total_loss = 0
    ns = 0
    for data in tqdm_obj:
        (clip_id, features, query_emb, starts, ends, is_ans) = data
        if features[0] is None: # TODO
            continue# TODO
        # print(clip_id, features.shape, query_emb.shape, starts.shape, ends.shape, is_ans.shape)
        ns += 1
        features = features.to(args.device)
        query_emb = query_emb.to(args.device)
        starts = starts.to(args.device)
        ends = ends.to(args.device)
        is_ans = is_ans.to(args.device)

        input_features = torch.cat((features, query_emb), dim=1)
        pred = model(input_features)
        loss = model_loss(pred, starts, ends, is_ans)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # Logging
        total_loss += loss.detach().cpu().item()
        n_iter = epoch * len(dataloader) + iter
        # if n_iter % args.log_interval == 0:
        writer.add_scalar('Loss/train', loss.detach().cpu().item(), n_iter)
            # wandb.log({"loss": loss.detach().cpu().item()}, step=n_iter)
        iter += 1

        # end train loop
    print("Train Examples = ", ns)

    return total_loss

def test(model, dataloader, model_loss, args, writer, epoch):
    model.eval()
    qa_pipeline = pipeline("question-answering")
    tqdm_obj = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Epoch %3d / %3d" % (epoch + 1, args.num_epochs),
            )
    iter = 0
    total_loss = 0
    records = []
    ns = 0
    for i, data in enumerate(tqdm_obj):
        (clip_id, features, query_emb, starts, ends, is_ans) = data
        if features[0] is None: # TODO
            continue# TODO
        ns+=1
        features = features.to(args.device)
        query_emb = query_emb.to(args.device)
        starts = starts.to(args.device)
        ends = ends.to(args.device)
        is_ans = is_ans.to(args.device)

        with torch.no_grad():
            input_features = torch.cat((features, query_emb), dim=1)
            pred = model(input_features)
            loss = model_loss(pred, starts, ends, is_ans)

        # infer
        s, e, scores = infer_from_model(pred, args.topk, qa_pipeline)
        records.append({"sample_id": int(i), 
                        "clip_id": str(clip_id),
                        "start": list([int(x) for x in s]), 
                        "end": list([int(x) for x in e]), 
                        "score": list([float(x) for x in scores]),
                        "GT_starts": int(np.where(starts.cpu().numpy() == 1)[0][0]),
                        "GT_ends": int(np.where(ends.cpu().numpy() == 1)[0][0]),
                        "Loss": float(loss.cpu().item())})

        # Logging
        total_loss += loss.cpu().item()
        n_iter = epoch * len(dataloader) + iter
        writer.add_scalar('Loss/val', loss.cpu().item(), n_iter)
        iter += 1

        # end val loop
    print("Test/Val Examples = ", ns)
    return total_loss, records

def initialise_wandb(args,model):
    wandb.init(project="ego4d", config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "hidden_size": args.hidden_size
    })
    wandb.watch(model)

if __name__ == "__main__":
    args = parse_args()
    train_loader, val_loader, test_loader, video_feature_size, query_feature_size = get_dataloader(args)
    args.embedding_dim = video_feature_size + query_feature_size
    model, model_loss, optimizer = init_model(args)
    # initialise_wandb(args,model)
    writer = SummaryWriter()

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, model_loss, optimizer, args, writer, epoch)
        val_loss, records = test(model, val_loader, model_loss, args, writer, epoch)
        if not os.path.exists("output/records"):
            os.makedirs("output/records")
        with open(f"output/records/records_{epoch}.json", "w") as f:
            json.dump(records, f, indent=4)

        #evaluate
        #test if better results
        test_loss, records = test(model, test_loader, model_loss, args, writer, epoch)
        with open(f"output/records/records_test_{epoch}.json", "w") as f:
            json.dump(records, f, indent=4)