from collections import defaultdict
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import RobertaTokenizer, RobertaModel
import json
import yaml
import argparse
import math
import enum
import os
import re
from tqdm import tqdm
import random
import numpy as np 

from model.data_utils import load_pickle, save_pickle

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
        self.modalities = modalities
        self.video_feature_size, self.audio_feature_size, self.query_feature_size = None, None, None

        if self.parsed_args.save_or_load and not self.parsed_args.update and self.parsed_args.save_or_load_path is not None:
            if os.path.exists(f'{self.parsed_args.save_or_load_path}_{self.split}.pkl'): #load
                print(f"Loading datafile from {self.parsed_args.save_or_load_path}_{self.split}.pkl")
                save_data = load_pickle(f'{self.parsed_args.save_or_load_path}_{self.split}.pkl')
                self.idx_frames = save_data['idx_frames']
                self.idx_sample_query = save_data['idx_sample_query']
                self.narrationCount = save_data['narrationCount']
                self.sample_query_map = save_data['sample_query_map']
                self.narrationData = save_data['narrationData'] 

                if self.parsed_args.input_type == 'multi_narration':
                    self._apply_filter_with_multi_narr(self.parsed_args)
                else:
                    self._apply_filter(self.parsed_args)
                print("Done Applying filters...")

                print(f"#{self.split} Frames: {self.idx_frames}")
                print(f"#{self.split} Sample: {self.idx_sample_query}")
                print(f"#{self.split} Narrations: {self.narrationCount}")
                print(f"#{self.split} Final Frames after filters: {self.idx_frames_filter}")
                print(f"#{self.split} Final Sample after filters: {len(self.data)}")
                
                for _idx in range(len(self.data)):    
                    _, video_features, audio_features, query_features, _, _, _ = self[_idx]
                    if video_features is not None:
                        self.video_feature_size = video_features[0].shape[-1]
                    if audio_features is not None:
                        self.audio_feature_size = audio_features[0][0].shape[-1]
                    if query_features is not None:
                        self.query_feature_size = query_features[0][0].shape[-1]

                    if self.video_feature_size is not None and self.audio_feature_size is not None and self.query_feature_size is not None:
                        break

                return
        
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
        if self.parsed_args.input_type == 'multi_narration':
            self._apply_filter_with_multi_narr(self.parsed_args)
        else:
            self._apply_filter(self.parsed_args)
        print("Done Applying filters...")

        print(f"#{self.split} Frames: {self.idx_frames}")
        print(f"#{self.split} Sample: {self.idx_sample_query}")
        print(f"#{self.split} Narrations: {self.narrationCount}")
        print(f"#{self.split} Final Frames after filters: {self.idx_frames_filter}")
        print(f"#{self.split} Final Sample after filters: {len(self.data)}")

        for _idx in range(len(self.data)):
            _, video_features, audio_features, query_features, _, _, _ = self[_idx]
            if video_features is not None:
                self.video_feature_size = video_features[0].shape[-1]
            if audio_features is not None:
                self.audio_feature_size = audio_features[0][0].shape[-1]
            if query_features is not None:
                self.query_feature_size = query_features[0].shape[-1]
            if self.video_feature_size is not None and self.audio_feature_size is not None and self.query_feature_size is not None:
                break

        # save it to a file if path given
        if self.parsed_args.save_or_load_path is not None and (self.parsed_args.save_or_load or self.parsed_args.update):
            self.save_data(f'{self.parsed_args.save_or_load_path}_{self.split}.pkl')
        
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
        narration_data = record['NarrationData']

        if self.parsed_args.input_type == 'multi_narration':
            start, end = record['ts_start'], record['ts_end']
            if self.modalities is not None:
                if (Modal._Video in self.modalities):
                    if video_path is not None:
                        video_features = torch.load(video_path)[start:end]
                
                if (Modal._Audio in self.modalities):
                    if audio_path is not None:
                        audio_features = torch.load(audio_path)[start:end]
            else:
                if video_path is not None:
                    video_features = torch.load(video_path)[start:end]
                if audio_path is not None:
                    audio_features = torch.load(audio_path)[start:end]
            
            narrFeatures = []
            for i in narration_data:
                narrFeatures.append(i['NarrationFeature'])
            return _vid, video_features, audio_features, narrFeatures, narration_data, start, end
        else:
            narr_feature_timestamp = narration_data['feature_frame_timestamp']
            start = 0 
            end = 0

            while True:   
                start = max(0, narr_feature_timestamp - self.parsed_args.input_length)
                start = random.randint(start, narr_feature_timestamp)
                end = min(start + self.parsed_args.input_length, record['total_feature_vector']-1)
                if start <= narr_feature_timestamp and start >= 0 and end <= record['total_feature_vector']-1 and end - start <= self.parsed_args.input_length and start < end and end > narr_feature_timestamp:
                    break
            
            if self.modalities is not None:
                if (Modal._Video in self.modalities):
                    if video_path is not None:
                        video_features = torch.load(video_path)[start:end]
                
                if (Modal._Audio in self.modalities):
                    if audio_path is not None:
                        audio_features = torch.load(audio_path)[start:end]
            else:
                if video_path is not None:
                    video_features = torch.load(video_path)[start:end]
                if audio_path is not None:
                    audio_features = torch.load(audio_path)[start:end]
            narration_data['ts_start'] = start
            narration_data['ts_end'] = end

            return _vid, video_features, audio_features, record['NarrationFeature'], narration_data, None, None

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
        self.narrationCount = 0
        for vid, values in tqdm(jsonData.items()):
            videoNarrData = defaultdict(list)
            if 'narration_pass_1' in values:
                for i in values['narration_pass_1']['narrations']:
                    _text = re.sub('(#[a-zA-Z]*)', '', i['narration_text']).strip()
                    i['narration_text_simplified'] = _text
                    i['feature_frame_timestamp'] = self._get_nearest_video_frame(i['timestamp_sec'], math.floor)
                    videoNarrData[i['feature_frame_timestamp']].append(i)
                    self.narrationCount +=1

            if 'narration_pass_2' in values:
                for i in values['narration_pass_2']['narrations']:
                    _text = re.sub('(#[a-zA-Z]*)', '', i['narration_text']).strip()
                    i['narration_text_simplified'] = _text
                    i['feature_frame_timestamp'] = self._get_nearest_video_frame(i['timestamp_sec'], math.floor)
                    videoNarrData[i['feature_frame_timestamp']].append(i)
                    self.narrationCount +=1

            if len(videoNarrData.keys()) != 0:
                self.narrationData[vid] = videoNarrData

    def _process_data(self, jsonData):
        """Process the entire json"""
        print("Processing ego4D json data...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'{device}')
        print(f'Model: {self.parsed_args.wordEmbedding_model}')
        tokenizer = RobertaTokenizer.from_pretrained(self.parsed_args.wordEmbedding_model)
        model = RobertaModel.from_pretrained(self.parsed_args.wordEmbedding_model, output_hidden_states = True ).to(device) # Whether the model returns all hidden-states.
        model.eval()

        tqdm_obj = tqdm(
                enumerate(jsonData['videos']),
                total=len(jsonData['videos']),
                desc="Videos",
            )
        for idx, data in tqdm_obj:

            _vid = data['video_uid']
            _narr = None
            
            clip_path = os.path.join(self.videos_path, _vid+'.pt')
            if not os.path.exists(clip_path):
                clip_path = None

            audio_path = os.path.join(self.audio_path, _vid+'.pt')
            if not os.path.exists(audio_path):
                audio_path = None
            print(audio_path)
                
            totalFrameDuration = self._get_nearest_video_frame(data['duration_sec'], math.floor)+1
                
            if _vid in self.narrationData:
                _narr = self.narrationData[_vid]
                for frameNo, narrs in _narr.items():
                    for j, narrData in enumerate(narrs):
                        _text = narrData['narration_text_simplified']
                        inputText = tokenizer(_text, return_tensors='pt').to(device)
                        with torch.no_grad():
                            _narrFeatures = model(**inputText).last_hidden_state.to('cpu')
                        record = {
                            "video_id": _vid,
                            "duration_sec": data['duration_sec'],
                            "total_feature_vector": totalFrameDuration,
                            "video_path": clip_path,
                            "audio_path": audio_path,
                            "NarrationData": narrData,
                            "NarrationFeature": _narrFeatures
                        }
                        self.sample_query_map[self.idx_sample_query] = record
                        self.idx_sample_query += 1
                        self.idx_frames += record['total_feature_vector']

    def _apply_filter_with_multi_narr(self, args):
        """Apply filters with multiple narration"""
        maxF = 10000000000
        minF = 1
        if args.max_frames != None:
            maxF = args.max_frames
        if args.min_frames != None:
            minF = args.min_frames
        tempRecords = defaultdict(list)
        for i, record in self.sample_query_map.items():
            if (record['total_feature_vector']-1) == record['NarrationData']['feature_frame_timestamp']:
                continue

            if self.modalities is not None:
                if (Modal._Video in self.modalities):
                    if  (record['video_path'] == None):
                        continue
                if (Modal._Audio in self.modalities):
                    if  (record['audio_path'] == None):
                        continue

            if record['total_feature_vector'] >= minF and record['total_feature_vector'] <= maxF:
                tempRecords[record['video_id']].append(record)
        for _vid, data in tempRecords.items():
            dataTemp = tempRecords[_vid][0]
            
            tempDict = defaultdict(list)
            for d in data:
                narr = d['NarrationData']
                narr['NarrationFeature'] = d['NarrationFeature']
                input_idx = narr['feature_frame_timestamp']//args.input_frames
                tempDict[input_idx].append(narr)
            for i, d in tempDict.items():
                if(len(d) == 0):
                    continue
                record = {
                    'video_id':dataTemp['video_id'],
                    'duration_sec':dataTemp['duration_sec'],
                    'total_feature_vector': dataTemp['total_feature_vector']-1,
                    'video_path': dataTemp['video_path'],
                    'audio_path': dataTemp['audio_path'],
                    'NarrationData': d,
                }
                if (record['total_feature_vector'] - (i*args.input_frames)) > args.input_frames-1:
                    self.idx_frames_filter += args.input_frames
                    record['ts_start'] = (i*args.input_frames)
                    record['ts_end'] = ((i+1)*args.input_frames)
                else:
                    self.idx_frames_filter += record['total_feature_vector'] - (i*args.input_frames)
                    record['ts_start'] = (i*args.input_frames)
                    record['ts_end'] = (record['total_feature_vector'])
                self.data.append(record)

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
            if (record['total_feature_vector']-1) == record['NarrationData']['feature_frame_timestamp']:
                continue

            if self.modalities is not None:
                if (Modal._Video in self.modalities):
                    if  (record['video_path'] == None):
                        continue
                if (Modal._Audio in self.modalities):
                    if  (record['audio_path'] == None):
                        continue

            if record['total_feature_vector'] >= minF and record['total_feature_vector'] <= maxF:
                self.data.append(record)
                self.idx_frames_filter += record['total_feature_vector']


def get_loader(dataset, batch_size, type=None):
    if type == "multi_narration":
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fu_for_multiNarr,
        )
    else:
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_collate_fn,
        )
    return train_loader

def collate_fu_for_multiNarr(batch):
    """
    Collate function for data loading.
    """
    _vid, video_features, audio_features, query_features, data, start, _ = zip(*batch)
    query_feat = query_features[0][0]
    num_queries = 1
    for q in query_features[0][1:]:
        if (query_feat.shape[1] + q.shape[1] + num_queries) >= 1050:
            break
        query_feat = torch.cat((query_feat, q), 1)   
        num_queries += 1 

    video_features = torch.stack(video_features)
    default_audio = torch.zeros(video_features[0].shape[0],49, 1024).to(video_features)
    audio_features = [torch.mean(x,dim=1) if x is not None else default_audio for x in audio_features]
    audio_features = torch.stack(audio_features)
    
    query_frame_numbers = torch.from_numpy(np.array([ d['feature_frame_timestamp'] - start[0]  for d in data[0][:num_queries]])).to(torch.float)

    return _vid, video_features, audio_features, query_features[0][:num_queries], query_feat.shape[1], query_frame_numbers, data[0][:num_queries]

def train_collate_fn(batch):
    """
    Collate function for data loading.
    """
    _vid, video_features, audio_features, query_features, data = zip(*batch)
    video_features = torch.stack(video_features)

    default_audio = torch.zeros(video_features[0].shape[0],49, 1024).to(video_features)
    audio_features = [torch.mean(x,dim=1) if x is not None else default_audio for x in audio_features]
    audio_features = torch.stack(audio_features)

    #get only CLS embedding for query
    query_features = torch.stack([query_features[0][0]])
    gtruth = np.zeros(video_features[0].shape[0])
    gtruth[data[0]['feature_frame_timestamp'] - data[0]['ts_start']] = 1
    gtruth = torch.from_numpy(gtruth).to(torch.float)

    return _vid, video_features, audio_features, query_features, gtruth, data
