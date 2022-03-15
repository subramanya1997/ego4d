from copy import deepcopy
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BertModel
import json
import yaml
import argparse

class MEMEDataLoader(Dataset):

    def __init__(self, json_path=None, videos_path=None, narrations_path=None, audio_path=None, config_file="dataset_config.yaml"):
        self.parsed_args = self._load_config(config_file)
        self.videos_path = videos_path
        self.narrations_path = narrations_path
        self.audio_path = audio_path
        self.ego4d_json = None
        self.narrations_json = None
        if json_path is not None:
            self.ego4d_json = self.load_json_data(json_path)
        if narrations_path is not None:
            self.narrations_json = self.load_json_data(narrations_path)
        
    def __len__(self):
        if self.ego4d_json == None:
            return 0
        return len(self.ego4d_json['videos'])

    def load_json_data(self, json_path):
        with open(json_path,'r') as f:
            json_data = f.read()
        return json.loads(json_data)

        
    def get_video_json(self, video_id):
        data_item = {}
        if self.ego4d_json == None:
            return None
        videos_json = self.ego4d_json['videos']
        vid_data = next(filter(lambda v : v['video_uid'] == video_id, videos_json), None)
        data_item.update(vid_data)
        if self.narrations_json is not None and video_id in self.narrations_json:
            narrations_pass = list(self.narrations_json[video_id].keys())[0]
            narrations_text = self.narrations_json[video_id][narrations_pass]['narrations']
            data_item['narrations'] = narrations_text
        return data_item
        
    def get_video_features(self, video_id):
        return

    def get_audio_features(self, video_id):
        return 

    def get_text_features(self, data_item):
        if not ('narrations' in data_item):
            return None
        narrations_text = data_item['narrations']
        # use cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # text features models
        tokenizer = BertTokenizer.from_pretrained(self.parsed_args.wordEmbedding_model)
        model = BertModel.from_pretrained(self.parsed_args.wordEmbedding_model, output_hidden_states = True ).to(device) # Whether the model returns all hidden-states.
        model.eval()  
        narration_features = []
        for narration_obj in narrations_text:
            _query = narration_obj['narration_text']
            input = tokenizer(_query, return_tensors='pt').to(device)
            _text_features = None
            with torch.no_grad():
                 _text_features = model(**input).last_hidden_state.to('cpu')
            feature_obj = deepcopy(narration_obj)
            feature_obj.pop('narration_text')
            feature_obj['text_features'] = _text_features
            narration_features.append(feature_obj)
        return narration_features

    def __get_item__(self, video_id):
        data_item = self.get_video_json(video_id)
        if data_item is not None:
            narr_features = self.get_text_features(data_item)
            vid_features = self.get_video_features(video_id)
            audio_features = self.get_audio_features(video_id)
            return data_item, vid_features, narr_features, audio_features
        return None

    
    def _get_nearest_audio_frame(self, time, floor_or_ceil=None):
        """Obtain the nearest frame for a given time, audio fps, and feature window."""
        return floor_or_ceil(int(time * self.parsed_args.audio_fps / self.parsed_args.audio_window_size))

    def _load_config(self, path):
        print("Loading Config...")
        parser = argparse.ArgumentParser(description=__doc__)
        parsed_args = parser.parse_args([])
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if key not in parsed_args.__dict__ or parsed_args.__dict__[key] is None:
                parsed_args.__dict__[key] = value
        return parsed_args



