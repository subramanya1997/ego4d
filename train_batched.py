import os
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
from transformers import pipeline

from model.meme import MEME
from model.meme_loss import MEME_LOSS
from model.utils import fix_seed, make_windows
from utils.metrics import decode_candidate_clips, get_best_segment, multi_infer, classification_metrics
from utils.evaluate_records import evaluate_predicted_records
from utils.data_processing import Ego4d_NLQ, get_train_loader, get_test_loader, Modal

ISSUE_CIDS = {}
# os.environ['TRANSFORMERS_CACHE'] = '/work/shantanuagar_umass_edu/ego4d/meme/ego4d/cache/'
torch.multiprocessing.set_start_method('spawn')
torch.multiprocessing.set_sharing_strategy('file_system')
'''
https://stackoverflow.com/a/31347222/4706073
'''
def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", default='config.yaml'
    )
    parser.add_argument("--max-len", help="maximum length of answer clip", type=int, default=None)
    parser.add_argument("--force-cpu", help="enforce cpu computation", type=int, default=None)
    parser.add_argument("-r", "--record-path", help="path for saving records", type=str, default='output/records/')
    parser.add_argument("-p", "--prefix", help="prefix for this run", type=str, default='joint')
    parser.add_argument("-l", "--loss-type", help="loss type to use", type=str, default='joint_loss')
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.0037)
    parser.add_argument("-w", "--wandb-name", \
                        help="wandb name for this run, defaults to random names by wandb",\
                        type=str, default=None)
    parser.add_argument("--model-save-path", help="path of the directory with model checkpoint", type=str, default=None)
    parser.add_argument("--load-path", help="path of the directory with model checkpoint that you want to load", type=str, default=None)
    parser.add_argument("--loss_weight", help="loss weight", type=float, default=0.9838)
    parser.add_argument("--loss_weight2", help="loss weight for balancing 2 tasks", type=float, default=0.9838)
    parser.add_argument("--loss_weight3", help="loss weight for balancing 2 tasks", type=float, default=0.5)
    parser.add_argument("--clip_window", help="clip_window", type=int, default=None)
    parser.add_argument("--num_epochs", help="epochs", type=int, default=None)
    add_bool_arg(parser, 'resume', default=False)
    add_bool_arg(parser, 'audio', default=True)
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
    
    # set best checkpoint and last checkpoint paths
    if parsed_args.model_save_path is None:
        parsed_args.model_save_path = "./output/models/"
    parsed_args.best_model_path = os.path.join(parsed_args.model_save_path, f"{parsed_args.prefix}_{parsed_args.wandb_name}_best.pth")
    parsed_args.last_model_path = os.path.join(parsed_args.model_save_path, f"{parsed_args.prefix}_{parsed_args.wandb_name}_last.pth")

    if parsed_args.load_path is None:
        parsed_args.load_path = parsed_args.last_model_path

    if not parsed_args.force_cpu:
        parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Parsed Arguments are - ", parsed_args)

    return parsed_args

def save_checkpoint(model, optimizer, epoch, loss, mIoU, path):
    torch.save({ # Save our checkpoint loc
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'mIoU': mIoU
            }, path)
    wandb.save(path)

def load_checkpoint(model, optimizer, path):
    # wandb.restore(path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    mIoU = checkpoint['mIoU']
    print(f"Loaded checkpoint from {path}")
    return model, optimizer, epoch, loss, mIoU

def get_dataloader(args):
    print("Loading data")
    train_nlq = Ego4d_NLQ(args.input_train_split, modalities=None, split="train", save_or_load_path=f"{args.dataloader_cache_path}/final_train.pkl", config_file = args.dataloader_config)
    val_nlq = Ego4d_NLQ(args.input_val_split, modalities=None, split="val", save_or_load_path=f"{args.dataloader_cache_path}/final_val.pkl", config_file = args.dataloader_config)
    test_nlq = Ego4d_NLQ(args.input_test_split, modalities=None, split="test", save_or_load_path=f"{args.dataloader_cache_path}/final_test.pkl", config_file = args.dataloader_config)

    train_loader = get_train_loader(train_nlq, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = get_test_loader(val_nlq, batch_size=args.batch_size)
    test_loader = get_test_loader(test_nlq, batch_size=args.batch_size)

    args.video_feature_size = train_nlq.video_feature_size if train_nlq.video_feature_size is not None else 0
    args.query_feature_size = train_nlq.query_feature_size if train_nlq.query_feature_size is not None else 0
    args.audio_feature_size = train_nlq.audio_feature_size if train_nlq.audio_feature_size is not None else 0

    print("Finished loading data")
    return args, train_loader, val_loader, test_loader, val_nlq, test_nlq

def init_model(args):
    print("Initializing model")
    model = MEME(args)
    model.to(args.device)
    parallel_model = torch.nn.DataParallel(model)

    model_loss = MEME_LOSS(args)
    model_loss.to(args.device)

    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    #scheduler = ConstantLR(optimizer, factor=0.95,total_iters=args.num_epochs*args.num_batches)
    print("Finished initializing model")

    return model, model_loss, optimizer, parallel_model

def infer_from_model(pred, topk, qa_pipeline, lens = None):
    # start = pred[:, 0].cpu().numpy()
    # end = pred[:, 1].cpu().numpy()
    # max_len = args.max_len
    b,l,c = pred.shape
    pred_ = pred[:,1:].reshape(-1,c)
    pred_ = pred_[:lens,:]
    pred_p = torch.nn.functional.softmax(pred_, dim=0).cpu().numpy()
    s, e, scores = decode_candidate_clips(qa_pipeline, pred_p[:,2], pred_p[:,3], topk, 50)
    # s, e, scores = get_best_segment(pred[:,1:,-1].cpu().numpy(), topk)
    # pred_p = torch.nn.functional.softmax(pred, dim=-1)
    # s, e, scores = get_best_segment_improved(pred_p.cpu().numpy(), topk)
    # pred_p = torch.nn.functional.softmax(pred, dim=-1)
    # s, e, scores = get_best_scoring_segment(pred_p.cpu().numpy(), topk)

    # s, e, scores = multi_infer(pred.cpu().numpy(), topk)
    
    return s, e, scores

def get_modalities(args):
    modals = [Modal._Video,Modal._Transcript]
    if args.audio:
        modals.append(Modal._Audio)
    return modals

def process_data(data, device):
    (sample_id, clip_id, features, audio_features, query_emb, starts, ends, is_ans, info, clip_lengths, query_lengths, offset, center_idx) = data
    features = features.to(device)
    audio_features = audio_features.to(device)
    query_emb = query_emb.to(device)
    starts = starts.to(device)
    ends = ends.to(device)
    is_ans = is_ans.to(device)
    clip_lengths = clip_lengths.to(device)
    query_lengths = query_lengths.to(device)
    offset = offset.to(device)
    center_idx = center_idx.to(device)
    return sample_id, clip_id, features, audio_features, query_emb, starts, ends, is_ans, info, clip_lengths, query_lengths, offset, center_idx

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
        data = process_data(data, args.device)
        (_, _, features, audio_features, query_emb, starts, ends, is_ans, _, clip_lengths, query_lengths, _, center_idx) = data
       
        pred, loss = model(features, query_emb, audio_features, clip_lengths = clip_lengths, \
                            query_lengths = query_lengths, center_idx = center_idx, is_ans = is_ans)
        # loss = model_loss(pred, starts, ends, is_ans, loss_type = args.loss_type)
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # Logging
        total_loss += loss.detach().cpu().item()
        n_iter = epoch * len(dataloader) + iter
        # if n_iter % args.log_interval == 0:
        writer.add_scalar('Loss/train', loss.detach().cpu().item(), n_iter)
        wandb.log({"batch loss/train": loss.detach().cpu().item()})
        iter += 1

        # end train loop
    total_loss /= len(dataloader)
    wandb.log({f"loss/train": total_loss})

    if epoch==0:
        ISSUE_CIDS['train'] = issue_cids
    return total_loss

def process_pred(pred):
    pred_ = torch.nn.functional.softmax(pred[:,:,:2], dim=-1)
    pred_ = pred_[:,:,1] > 0.5
    pred_ = pred_.to(torch.float32) 
    return pred_

def test_model(model, dataloader, model_loss, args, writer, epoch, Test = False):
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
    pred_ = []
    gt_ = []
    for i, data in enumerate(tqdm_obj):
        data = process_data(data, args.device)
        (sample_id, clip_id, features, audio_features, query_emb, starts, ends, is_ans, info, clip_lengths, query_lengths, offsets, center_idx) = data
    
        with torch.no_grad():
            pred, loss = model(features, query_emb, audio_features, clip_lengths = clip_lengths, \
                            query_lengths = query_lengths, center_idx = center_idx, is_ans = is_ans)
            loss = loss.mean()
            # loss = model_loss(pred, starts, ends, is_ans, loss_type = args.loss_type)
            
        # infer
        # pred = (batch size, max_len)
        for i, p in enumerate(pred):
            end_orig = info[i]['end_frame_idx']
            start_orig = info[i]['start_frame_idx']
            offset = offsets[i].cpu().numpy().item()
            pred_idx = np.argmax(p.detach().cpu().numpy(), axis=-1)[0]
            gt = center_idx.cpu().numpy()[i]
            pred_.append(np.abs(pred_idx-gt)<=5)
            s, e, scores = [], [], []
            for i in range(5):
                s.append(max(0,pred_idx-10*(i+1)))
                e.append(min(p.shape[-1],pred_idx+10*(i+1)))
                scores.append(1/(i+1))

            # print(sample_id, clip_id, torch.sum(ends))
            records.append({"sample_id": int(sample_id[0]), 
                            "clip_id": str(clip_id[0]),
                            "start": list([int(offset+x) for x in s]), 
                            "pred_center": int(pred_idx), 
                            "gt_center": int(gt), 
                            "end": list([int(offset+x) for x in e]), 
                            "score": list([float(x) for x in scores]),
                            "orig center": int(gt+offset),
                            "GT_starts": int(start_orig),
                            "GT_ends": int(end_orig),
                            "Frame length": int(info[i]['Frame length']),
                            "Loss": float(loss.cpu().item())})

        # Logging
        total_loss += loss.cpu().item()
        n_iter = epoch * len(dataloader) + iter
        split = "test" if Test else "val"
        writer.add_scalar(f'Loss/{split}', loss.cpu().item(), n_iter)
        wandb.log({f"batch loss/{split}": loss.cpu().item()})

        iter += 1

        # end val loop
    total_loss /= len(dataloader)
    split = "Test" if Test else "Val"
    wandb.log({f"loss/{split}": total_loss})
    split = "test" if Test else "val"
    print("Accuracy/acc", np.mean(pred_))
    wandb.log({f"{split}/acc": np.mean(pred_)})

    return total_loss, records

def initialise_wandb(args,model):
    wandb.init(project=f"ego4d_{args.prefix}", entity="ego4d-meme", config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "loss_weight": args.loss_weight,
    },resume = args.resume,settings=wandb.Settings(start_method="fork"))
    if args.wandb_name is not None:
        wandb.run.name = args.wandb_name
        wandb.run.save()

    wandb.watch(model)

def cache_records_and_evaluate(records, epoch, n_iter, args, nlq_data, writer, test=False):
    if not os.path.exists(args.record_path):
        os.makedirs(args.record_path)
    folder_name = f"{args.prefix}_{args.wandb_name}" if args.wandb_name is not None else args.prefix
    if not os.path.exists(os.path.join(args.record_path, folder_name)):
        os.makedirs(os.path.join(args.record_path, folder_name))
    file_name = f"{folder_name}/records_{epoch}.json" if not test else f"{folder_name}/records_test_{epoch}.json"
    with open(os.path.join(args.record_path, file_name), "w") as f:
        json.dump(records, f, indent=4)
    sys.stdout.flush()
    
    # gt_file = args.input_val_split if not test else args.input_test_split
    gt_file = args.input_val_split if not test else args.input_test_split
    _, mIoU, score_str, metric_dict = evaluate_predicted_records(records, epoch, gt_file, nlq_data)
    for metric, value in metric_dict.items():
        split = "val" if not test else "test"
        # writer.add_scalar(f"{metric}/{split}", value, epoch)
        wandb.log({f"{split}/{metric}": value})
        # wandb.run.summary[f"{metric}/{split}"] = value
    
    print(score_str)
    return mIoU


if __name__ == "__main__":
    fix_seed(42)
    args = parse_arguments()
    args, train_loader, val_loader, test_loader, val_nlq, test_nlq = get_dataloader(args)
    args.embedding_dim = args.video_feature_size + args.query_feature_size 
    if args.audio:
        args.embedding_dim += args.audio_feature_size

    model, model_loss, optimizer, parallel_model = init_model(args)
    initialise_wandb(args,model)
    writer = SummaryWriter()
    best_mIoU = 0
    start_epoch = 0

    if wandb.run.resumed or args.resume:
        model, optimizer, start_epoch, loss, best_mIoU = load_checkpoint(model, optimizer, args.load_path)
        print(f"Loaded checkpoint from {args.load_path} from epoch {start_epoch} with loss {loss}")

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train(parallel_model, train_loader, model_loss, optimizer, args, writer, epoch)
        val_loss, records = test_model(parallel_model, val_loader, model_loss, args, writer, epoch)
        val_mIoU = cache_records_and_evaluate(records, epoch, epoch * len(val_loader),args, val_nlq, writer)

        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            save_checkpoint(model, optimizer, epoch, val_loss, best_mIoU, args.best_model_path)

        save_checkpoint(model, optimizer, epoch, val_loss, best_mIoU, args.last_model_path)