from collections import defaultdict
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BertModel
import json
import yaml
import argparse
import math
import enum
import os
import re

import sys 
sys.path.append(r'../utils')

from utils.data_utils import load_pickle, save_pickle

class Modal(enum.Enum):
    _Video = 1
    _Audio = 2
    _Transcript = 3
    _IMU = 4
    _3D = 5

class MEMEDataLoader(Dataset):
    def __init__(self, json_path=None, modalities=None, split="train", config_file="dataset_config.yaml"):
        """Class for reading and visualizing for Self Supervised methods
        Args:
            json_path (str): location to data json (default: None)

        """
        self.parsed_args = self._load_config(config_file)
        print("Done Loading Config...")

        self.videos_path = self.parsed_args.video_features_path
        self.audio_path = self.parsed_args.audio_features_path
        self.narrations_path = self.parsed_args.narrations_path
        self.idx_frames = 0
        self.idx_sample_query = 0
        self.narrationCount = 0
        self.sample_query_map = {}
        self.narrationData = {}
        self.data = []
        self.split = split
        self.narrationCount_afterFilter = 0
        self.idx_frames_filter = 0
        
        assert (
            self.videos_path is not None
        ), "No Video Features Path Found - Please updated dataset_config_ssl.yaml"

        assert (
            self.audio_path is not None
        ), "No Audio Features Path Found - Please updated dataset_config_ssl.yaml"

        assert (
            self.narrations_path is not None
        ), "No Narrations json path Path Found - Please updated dataset_config_ssl.yaml"

        assert (
            json_path is not None
        ), "No json to load the data from..."

        rawjsonData = self._load_json(json_path)
        rawNarrationJsonData = self._load_json(self.narrations_path)

        self._process_narration(rawNarrationJsonData)
        self._process_data(rawjsonData)
        print("Done processing data...")
        self._apply_filter(self.parsed_args)
        print("Done Applying filters...")

        print(f"#{self.split} Frames: {self.idx_frames}")
        print(f"#{self.split} Videos: {self.idx_sample_query}")
        print(f"#{self.split} Narrations: {self.narrationCount}")
        print(f"#{self.split} Final Frames after filters: {self.idx_frames_filter}")
        print(f"#{self.split} Final Videos after filters: {len(self.data)}")
        print(f"#{self.split} Final Narrations after filters: {self.narrationCount_afterFilter}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """get Items from data

        """
        record = self.data[idx]
        _vid = record['video_id']
        video_path = record['video_path']
        audio_path = record['audio_path']

        video_features = None
        audio_features = None
        narration_data = None

        if _vid in self.narrationData:
            narration_data = self.narrationData[_vid]

        if self.modalities is not None:
            if (Modal._Video in self.modalities):
                if video_path is not None:
                    video_features = torch.load(video_path)
            
            if (Modal._Audio in self.modalities):
                if audio_path is not None:
                    audio_features = torch.load(audio_path)
        else:
            if video_path is not None:
                video_features = torch.load(video_path)
            if audio_path is not None:
                audio_features = torch.load(audio_path)

        return _vid, video_features, audio_features, narration_data

    def save_data(self, path):
        """Save data to path"""
        save_data = {}
        save_data['idx_frames'] = self.idx_frames
        save_data['idx_sample_query'] = self.idx_sample_query
        save_data['narrationCount'] = self.narrationCount
        save_data['sample_query_map'] = self.sample_query_map
        save_data['narrationData'] = self.narrationData

        save_pickle(save_data, path)


    def _load_config(self, path):
        print("Loading Config...")
        parser = argparse.ArgumentParser(description=__doc__)
        parsed_args = parser.parse_args([])
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if key not in parsed_args.__dict__ or parsed_args.__dict__[key] is None:
                if value == 'None':
                    parsed_args.__dict__[key] = None
                else:
                    parsed_args.__dict__[key] = value
        return parsed_args

    def _load_json(self, json_path):
        with open(json_path,'r') as f:
            json_data = f.read()
        return json.loads(json_data)

    def _get_nearest_video_frame(self, time, floor_or_ceil=None):
        """Obtain the nearest frame for a given time, video fps, and feature window."""
        return floor_or_ceil(int(time * self.parsed_args.video_fps / self.parsed_args.video_window_size))

    def _process_narration(self, jsonData):
        """Process Narration json"""
        print("Processing Narration data...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'{device}')
        tokenizer = BertTokenizer.from_pretrained(self.parsed_args.wordEmbedding_model)
        model = BertModel.from_pretrained(self.parsed_args.wordEmbedding_model, output_hidden_states = True ).to(device) # Whether the model returns all hidden-states.
        model.eval()

        for vid, values in jsonData.items():
            videoNarrData = defaultdict(list)
            narrCount = 0
            if 'narration_pass_1' in values:
                for i in values['narration_pass_1']['narrations']:
                    _text = re.sub('(#[a-zA-Z]*)', '', i['narration_text']).strip()
                    inputText = tokenizer(_text, return_tensors='pt').to(device)
                    with torch.no_grad():
                        i['textFeatures'] = model(**inputText).last_hidden_state.to('cpu')

                    videoNarrData[self._get_nearest_video_frame(i['timestamp_sec'], math.floor)].append(i)
                    narrCount+=1

            if 'narration_pass_2' in values:
                for i in values['narration_pass_2']['narrations']:
                    _text = re.sub('(#[a-zA-Z]*)', '', i['narration_text']).strip()
                    inputText = tokenizer(_text, return_tensors='pt').to(device)
                    with torch.no_grad():
                        i['textFeatures'] = model(**inputText).last_hidden_state.to('cpu')

                    videoNarrData[self._get_nearest_video_frame(i['timestamp_sec'], math.floor)].append(i)
                    narrCount+=1

            if len(videoNarrData) != 0:
                videoNarrData['Total'] = narrCount
                self.narrationData[vid] = videoNarrData
                self.narrationCount += narrCount

    def _process_data(self, jsonData):
        """Process the entire json"""
        print("Processing ego4D json data...")
        for i, data in enumerate(jsonData['videos']):

            _vid = data['video_uid']

            clip_path = os.path.join(self.videos_path, _vid+'.pt')
            if not os.path.exists(clip_path):
                clip_path = None

            audio_path = os.path.join(self.audio_path, _vid+'.pt')
            if not os.path.exists(audio_path):
                audio_path = None

            record = {
                "video_id": _vid,
                "duration_sec": data['duration_sec'],
                "total_feature_vector": self._get_nearest_video_frame(data['duration_sec'], math.floor)+1,
                "video_path": clip_path,
                "audio_path": audio_path,
            }
            self.sample_query_map[self.idx_sample_query] = record
            self.idx_sample_query += 1
            self.idx_frames += record['total_feature_vector']

    def _apply_filter(self, args):
        """Apply filters data"""
        print("Apply filters to data...")
        maxF = 10000000000
        minF = 1
        if args.max_frames != None:
            maxF = args.max_frames
        if args.min_frames != None:
            minF = args.min_frames
        for i, record in self.sample_query_map.items():
            if args.number_of_sample != None:
                if len(self.data) >= args.number_of_sample:
                    break
            if record['total_feature_vector'] >= minF and record['total_feature_vector'] <= maxF:
                self.data.append(record)
                self.idx_frames_filter += record['total_feature_vector']

                if record['video_id'] in self.narrationData:
                    self.narrationCount_afterFilter += self.narrationData[record['video_id']]['Total']