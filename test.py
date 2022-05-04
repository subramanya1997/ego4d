from utils.data_processing import Ego4d_NLQ, Modal

import argparse
from collections import defaultdict
import os

import torch

nql = Ego4d_NLQ('/scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_train.json', modalities=None, split="train", save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/final_roberta_train.pkl")
# print(nql.query_feature_size, nql.audio_feature_size, nql.video_feature_size)
# for i in range(nql.__len__()):
#     sample = nql[i]
nql = Ego4d_NLQ('/scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_val.json', modalities=None, split="val", save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/final_roberta_val.pkl")
# for i in range(nql.__len__()):
#     sample = nql[i]
nql = Ego4d_NLQ('/scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_test.json', modalities=None, split="test", save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/final_roberta_test.pkl")
# for i in range(nql.__len__()):
#     sample = nql[i]