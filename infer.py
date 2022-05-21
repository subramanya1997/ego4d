"""Main script to train/test models for Ego4D NLQ dataset.
"""
import argparse
import os

import numpy as np
import options
import torch
import torch.nn as nn
from model.MEME import build_optimizer_and_scheduler, MEME
from tqdm import tqdm
from exputils.data_gen import gen_or_load_dataset
from exputils.data_loader import get_test_loader, get_train_loader
from exputils.data_util import load_json, load_video_features, save_json
from exputils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)

def main(configs, parser):
    # set tensorflow configs
    set_th_config(configs.seed)

    # prepare or load dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset.get("n_chars", -1)
    configs.word_size = dataset.get("n_words", -1)

    # get train and test loader
    visual_features = load_video_features(
        os.path.join("/work/snagabhushan_umass_edu/dataset/", configs.task, configs.fv), configs.max_pos_len
    )
    # If video agnostic, randomize the video features.
    if configs.video_agnostic:
        visual_features = {
            key: np.random.rand(*val.shape) for key, val in visual_features.items()
        }
    train_loader = get_train_loader(
        dataset=dataset["train_set"], video_features=visual_features, configs=configs
    )
    train_eval_loader = get_test_loader(
        dataset=dataset["train_set"], video_features=visual_features, configs=configs
    )
    val_loader = (
        None
        if dataset["val_set"] is None
        else get_test_loader(dataset["val_set"], visual_features, configs)
    )
    test_loader = get_test_loader(
        dataset=dataset["test_set"], video_features=visual_features, configs=configs
    )
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)
    num_val_batches = 0 if val_loader is None else len(val_loader)
    num_test_batches = len(test_loader)

    # Device configuration
    cuda_str = "cuda" if configs.gpu_idx is None else "cuda:{}".format(configs.gpu_idx)
    device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")

    for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="evaluate {}".format("val"),
        ):

        print(vfeats.shape, vfeat_lens.shape, [ (i['sample_id'], i['vid'], i['s_ind'], i['e_ind'], i['s_time'], i['e_time']) for i in records])
        break

    # create model dir
    home_dir = os.path.join(
        configs.model_dir,
        "_".join(
            [
                configs.model_name,
                configs.task,
                configs.fv,
                str(configs.max_pos_len),
                configs.predictor,
            ]
        ),
    )
    if configs.suffix is not None:
        home_dir = home_dir + "_" + configs.suffix
    model_dir = os.path.join(f'{home_dir}_test2', "model")

    if not os.path.exists(model_dir):
        raise ValueError("No pre-trained weights exist")
    # load previous configs
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # build model
    model = MEME(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)

    # get last checkpoint file
    filename = get_last_checkpoint(model_dir, suffix="t7")
    model.load_state_dict(torch.load(filename))
    model.eval()
    result_save_path = filename.replace(".t7", "_test_result.json")
    results, mIoU, score_str = eval_test(
        model=model,
        data_loader=val_loader,
        device=device,
        mode="test",
        gt_json_path=configs.eval_gt_json,
        result_save_path=result_save_path,
    )
    print(results, mIoU, score_str)

if __name__ == "__main__":
    configs, parser = options.read_command_line()
    main(configs, parser)