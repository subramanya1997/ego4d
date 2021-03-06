import os
import json
import pickle
import torch

import numpy as np

from tqdm import tqdm
from config import Config
from collections import defaultdict
import enum

def readJsonFile(path="./dataset/tmp/ego4d.json"):
    with open(path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
        return jsonObject
    
def get_nearest_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, video fps, and feature window."""
    return floor_or_ceil(int(time * Config.VIDEO_FPS / Config.WINDOW_SIZE))

def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
        return data

def save_pickle(data, filename):
    with open(filename, mode="wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pad_seq(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_seq(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_seq(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_seq(
        sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length
    )
    sequence_length, _ = pad_seq(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def pad_video_seq(sequences, max_length=None):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length

def pad_2d_tensor_seq(sequences, pad_tok=None, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: x.shape[0], sequences))
    
    sequence_length = list(map(lambda x: x.shape[1], sequences))
    if max_length_2 is None:
        max_length_2 = max(sequence_length)

    for seq in sequences:
        padding_length = max_length_2 - seq.shape[1]
        padding_shape = [max_length, padding_length, seq.shape[2]]
        padding_seq = torch.zeros(padding_shape).to(seq)
        seq_ = torch.cat([seq, padding_seq], dim=1)
        sequence_padded.append(seq_)
    
    sequence_padded = torch.cat(sequence_padded, dim=0)
    return sequence_padded, sequence_length
