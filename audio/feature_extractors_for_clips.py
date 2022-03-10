import argparse

import torch

import os
from tqdm import tqdm
import json
from collections import defaultdict
import math

def get_nearest_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, video fps, and feature window."""
    return floor_or_ceil(int(time * 2))

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_all_audio_features_files(args):
    paths = defaultdict(str)
    for root, dirs, files in os.walk(args['input_audio_feature_directory']):
        for file in files:
            if file.endswith(".pt"):
                paths[file.split('.')[0]] = os.path.join(root, file)

    print("Number of audio files found in ",args['input_audio_feature_directory']," is ", len(paths.keys()))
    return paths

def getFeatures(args, paths):
    json_data = load_json(args['annotations_path'])
    for video_data in tqdm(json_data['videos']):
        _vid = video_data['video_uid']
        if (_vid in paths.keys()):
            audio_rep = torch.load(paths[_vid])
            for clip_data in video_data['clips']:
                _cid = clip_data['clip_uid']
                clip_path = os.path.join(args['audio_feature_save_directory'], _cid+'.pt')
                _vs_sec = get_nearest_frame(clip_data['video_start_sec'], math.floor)
                _ve_sec = get_nearest_frame(clip_data['video_end_sec'], math.ceil)
                _clip_audio_rep = audio_rep[_vs_sec:_ve_sec, :]
                torch.save(_clip_audio_rep, clip_path)
                #print(_cid, _vs_sec, _ve_sec, clip_data['video_start_sec'], clip_data['video_end_sec'], _clip_audio_rep.shape[0], _ve_sec-_vs_sec )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations_path", required=True, help="annotations_path",
    )
    parser.add_argument(
        "--input_audio_feature_directory", required=True, help="Path to audio features"
    )
    parser.add_argument(
        "--audio_feature_save_directory", required=True, help="Path to save clip audio features",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    paths = get_all_audio_features_files(parsed_args)
    getFeatures(args=parsed_args, paths=paths)
    