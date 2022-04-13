#import libs
import json
import os
import math
import yaml
import argparse
import enum
import random

import numpy as np

from re import L
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

#import custom functions
from utils.data_utils import load_pickle, save_pickle

from collections import defaultdict

class Modal(enum.Enum):
    _Video = 1
    _Audio = 2
    _Transcript = 3
    _IMU = 4
    _3D = 5

class Ego4d_NLQ(Dataset):
    def __init__(self, annotations_path, modalities = None, split="train", config_file="dataset_config.yaml", save_or_load_path=None, filter_vids = None):
        """Class for reading and visualizing annotations.
        Args:
            annotations_path (str): location of annotation file
            modalities (Modal): List of modalities to consider or None if you want to consider all of type Modal (default: None)
            split (str): train, val or test
            config_file (path): config file which has fields video_features_path, audio_features_path, video_fps, video_window_size, 
                                audio_fps, audio_window_size, wordEmbedding_model, number_of_sample, max_frames, min_frames, save_or_load
                                update
            save_or_load_path (str): path to load or save
            filter_vids (str): List of video ids to get clips from or None
        """
        self.parsed_args = self._load_config(config_file)
        print("Done Loading Config...")

        if self.parsed_args.save_or_load and not self.parsed_args.update and save_or_load_path is not None:
            if os.path.exists(save_or_load_path): #load
                print(f"Loading datafile from {save_or_load_path}")
                saved_data = load_pickle(save_or_load_path)
                print(f"Done loading datafile...")
                #about dataset
                self.version = saved_data['version']
                self.description = saved_data['description']
                #clip - video details
                self.all_clip_video_map = saved_data['all_clip_video_map']
                # split
                self.split = saved_data['split']
                #dataset info
                self.idx_counter = saved_data['idx_counter']
                self.data = []
                self.idx_sample_query = saved_data['idx_sample_query']
                self.sample_query_map = saved_data['sample_query_map']
                self.video_feature_size = saved_data['video_feature_size']
                self.audio_feature_size = 1024 # saved_data['audio_feature_size']
                self.query_feature_size = saved_data['query_feature_size']
                # self.default_audio_path = saved_data['default_audio_path']
                #models and options
                self.wordEmbedding_model =  self.parsed_args.wordEmbedding_model
                self.max_frames = self.parsed_args.max_frames if self.parsed_args.max_frames != 'None' else None
                self.min_frames = self.parsed_args.min_frames if self.parsed_args.min_frames != 'None' else None
                self.number_of_sample = self.parsed_args.number_of_sample if self.parsed_args.number_of_sample != 'None' else None
                self.video_features_path = self.parsed_args.video_features_path
                self.audio_features_path = self.parsed_args.audio_features_path
                self.modalities = set(modalities) if modalities is not None else None
                self.filter_vids = set(filter_vids) if filter_vids is not None else None
                # self.default_audio = self._load_default_audio(self.default_audio_path)

                self._get_final_dataset()
                print("Done processing data...")
                
                print(f"#{self.split} frames: {self.idx_counter}")
                print(f"#{self.split} clips: {self.idx_sample_query}")
                print(f"#{self.split} final clips after filters: {len(self.data)}")
                return

        raw_data = self._load_json(annotations_path)
        assert (
            type(raw_data) == dict
        ), "Annotation file format {} not supported.".format(type(raw_data))
        print("Done Loading Annotations...")
        #about dataset
        self.version = raw_data['version']
        self.description = raw_data['description']
        #clip - video details
        self.all_clip_video_map = {}
        # split
        self.split = split
        #dataset info
        self.idx_counter = 0
        self.data = []
        self.idx_sample_query = 0
        self.sample_query_map = {}
        self.video_feature_size = None
        self.audio_feature_size = None
        self.query_feature_size = None
        #models and options
        self.wordEmbedding_model = self.parsed_args.wordEmbedding_model
        self.max_frames = self.parsed_args.max_frames if self.parsed_args.max_frames != 'None' else None
        self.min_frames = self.parsed_args.min_frames if self.parsed_args.min_frames != 'None' else None
        self.number_of_sample = self.parsed_args.number_of_sample if self.parsed_args.number_of_sample != 'None' else None
        self.video_features_path = self.parsed_args.video_features_path 
        self.audio_features_path = self.parsed_args.audio_features_path
        self.modalities = set(modalities) if modalities is not None else None
        self.filter_vids = set(filter_vids) if filter_vids is not None else None
        # self.default_audio_path = self.parsed_args.default_audio_path
        
        assert (
            self.video_features_path is not None
        ), "No Video Features Path Found - Please updated dataset_config.yaml"

        assert (
            self.audio_features_path is not None
        ), "No Audio Features Path Found - Please updated dataset_config.yaml"
        
        #reformat data
        raw_data, clip_video_map = self._reformat_data(raw_data)
        self.all_clip_video_map.update(clip_video_map)
        print("Done reforming data...")
        
        #process dataset
        self._process_dataset(raw_data)
        print("Done processing data...")
        self._get_final_dataset()
        print("Done adding filters to data...")

        print(f"#{self.split} frames: {self.idx_counter}")
        print(f"#{self.split} clips: {self.idx_sample_query}")
        print(f"#{self.split} final clips after filters: {len(self.data)}")
        
        # self.default_audio = self._load_default_audio(self.default_audio_path)
        print("Done loading default audio...")
        
        # get feature sizes
        if self.idx_counter != 0:
            """ sample_id, clip_id, clip_features, audio_features, query_features, is_s, is_e, is_ans, frame_length """
            _, _, clip_feature, audio_features, query_features, _, _, _, _ = self[0]
            if clip_feature is not None:
                self.video_feature_size = clip_feature[0].shape[-1]
            if audio_features is not None:
                self.audio_feature_size = 1024 # audio_features[0][0].shape[-1]
            if query_features is not None:
                self.query_feature_size = query_features[0].shape[-1]
        else:
            print('No Data Loaded!')

        # save it to a file if path given
        if save_or_load_path is not None and (self.parsed_args.save_or_load or self.parsed_args.update):
            self.save_data(save_or_load_path)

    def __len__(self):
        """Length of the data"""
        return len(self.data)

    def __getitem__(self, idx):
        """ set item from data 
            Args:
                idx (int): index in self.data array (index in the filtered array)
            returns:
                sample_id(int), clip_id(str), clip_features(tensor or None), audio_features(tensor or None), 
                query_features(tensor or None), is_s(list(bool)), is_e(list(bool)), is_ans(list(bool)), 
                frame_length(list(int))
        """
        sample_id, sample_query = self.data[idx]
        s_v_idx, e_v_idx = sample_query["video_range"]
        s_a_idx, e_a_idx = sample_query["audio_range"]
        
        clip_path = sample_query['clip_path'].replace('scratch', 'work')
        clip_id = sample_query['clip_id']

        audio_path = sample_query['audio_path'].replace('scratch', 'work') if sample_query['audio_path'] is not None else None

        # clip_path = clip_path.replace('/scratch/snagabhushan_umass_edu/dataset/v1/clip_features/','data/saved_clip_features/')
        # audio_path = clip_path.replace('/scratch/snagabhushan_umass_edu/dataset/v1/clip_features/','/work/snagabhushan_umass_edu/dataset/v1/audio_features_from_video/')
        
        clip_features = None
        audio_features = None
        if self.modalities is not None:
            if (Modal._Video in self.modalities):
                if clip_path is not None:
                    clip_features = torch.load(clip_path)
                    clip_features = clip_features[ s_v_idx : e_v_idx , : ]

            if (Modal._Audio in self.modalities):
                if audio_path is not None:
                    audio_features = torch.load(audio_path)
                    audio_features = audio_features[ s_a_idx : e_a_idx , : ]
        else:
            if clip_path is not None:
                clip_features = torch.load(clip_path)
                clip_features = clip_features[ s_v_idx : e_v_idx , : ]

            if audio_path is not None:
                audio_features = torch.load(audio_path)
                audio_features = audio_features[ s_a_idx : e_a_idx , : ]

        query_features = sample_query['query_features']
        if query_features is not None:
            query_features = [query_features for item in range(s_v_idx, e_v_idx)]
        
        is_s = [ (sample_query['s_video_frame'] == i) for i in range(s_v_idx, e_v_idx)]
        is_e = [ (sample_query['e_video_frame'] == i) for i in range(s_v_idx, e_v_idx)]
        is_ans = [ (sample_query['s_video_frame'] <= i and sample_query['e_video_frame'] >= i) for i in range(s_v_idx, e_v_idx)]
        frame_length = sample_query['video_frame_length']#[ sample_query['video_frame_length'] for i in range(s_v_idx, e_v_idx)]

        info = {"Video Feature Size": self.video_feature_size,
                "Audio Feature Size": self.audio_feature_size,
                "Query Feature Size": self.query_feature_size,
                "Frame length": frame_length,
                # "default_audio": self.default_audio
                }
        
        return sample_id, clip_id, clip_features, audio_features, query_features, is_s, is_e, is_ans, info

    def _get_nearest_video_frame(self, time, floor_or_ceil=None):
        """Obtain the nearest frame for a given time, video fps, and feature window."""
        return floor_or_ceil(int(time * self.parsed_args.video_fps / self.parsed_args.video_window_size))

    def _get_nearest_audio_frame(self, time, floor_or_ceil=None):
        """Obtain the nearest frame for a given time, audio fps, and feature window."""
        return floor_or_ceil(int(time * self.parsed_args.audio_fps / self.parsed_args.audio_window_size))

    def getfromidx(self, idx):
        """ Has to be updated """
        sample_map = self.sample_query_map[idx]
        values = {
            "annotation_uid": sample_map["annotation_uid"],
            "clip_id": sample_map["clip_id"],
            "query_idx": sample_map["query_idx"],
            "query": sample_map["query"],
            "s_time": sample_map["exact_s_time"],
            "e_time": sample_map["exact_e_time"],
        }
        return values
    
    def save_data(self, path):
        """Save data to path"""
        saved_data = {}
        #about dataset
        saved_data['version'] = self.version
        saved_data['description'] = self.description 
        #clip - video details
        saved_data['all_clip_video_map'] = self.all_clip_video_map
        # split
        saved_data['split'] = self.split
        #dataset info
        saved_data['idx_counter'] = self.idx_counter
        saved_data['idx_sample_query'] = self.idx_sample_query
        saved_data['sample_query_map'] = self.sample_query_map
        saved_data['video_feature_size'] = self.video_feature_size
        saved_data['audio_feature_size'] = self.audio_feature_size 
        saved_data['query_feature_size'] = self.query_feature_size
        # saved_data['default_audio_path'] = self.default_audio_path

        #if not os.path.exists(path): #save create folder
            #pass
        save_pickle(saved_data, path)

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

    def _load_json(self, path):
        """Load annoations json"""
        print("Loading annotations...")
        with open(path, "r") as f:
            return json.load(f)

    def _process_question(self, question):
        """Process the question to make it canonical."""
        return question.strip(" ").strip("?").lower() + "?"

    def _reformat_data(self, split_data):
        """Convert the format from JSON files.
        fps, num_frames, timestamps, sentences, exact_times,
        annotation_uids, query_idx.
        """
        print("Reforming data...")
        formatted_data = {}
        clip_video_map = {}
        for video_datum in split_data["videos"]:
            for clip_datum in video_datum["clips"]:
                clip_uid = clip_datum["clip_uid"]
                clip_video_map[clip_uid] = (
                    video_datum["video_uid"],
                    clip_datum["video_start_sec"],
                    clip_datum["video_end_sec"],
                    clip_datum["clip_start_sec"],
                    clip_datum["clip_end_sec"],
                )
                num_frames = self._get_nearest_video_frame(clip_datum["video_end_sec"], math.ceil) - self._get_nearest_video_frame(clip_datum["video_start_sec"], math.floor) 
                a_num_frames = self._get_nearest_audio_frame(clip_datum["video_end_sec"], math.ceil) - self._get_nearest_audio_frame(clip_datum["video_start_sec"], math.floor)
                new_dict = {
                    "fps": self.parsed_args.video_fps / self.parsed_args.video_window_size,
                    "num_frames": num_frames,
                    "a_num_frames": a_num_frames,
                    "timestamps": [],
                    "a_timestamps": [],
                    "exact_times": [],
                    "sentences": [],
                    "query_templates": [],
                    "annotation_uids": [],
                    "query_idx": [],
                }

                for ann_datum in clip_datum["annotations"]:
                    for index, datum in enumerate(ann_datum["language_queries"]):
                        
                        start_time = float(datum["clip_start_sec"])
                        end_time = float(datum["clip_end_sec"])

                        if "query" not in datum or not datum["query"]:
                            continue
                        new_dict["sentences"].append(self._process_question(datum["query"]))
                        new_dict["query_templates"].append(datum["template"])
                        new_dict["annotation_uids"].append(ann_datum["annotation_uid"])
                        new_dict["query_idx"].append(index)
                        new_dict["exact_times"].append([start_time, end_time]),
                        new_dict["timestamps"].append(
                            [
                                self._get_nearest_video_frame(start_time, math.floor),
                                self._get_nearest_video_frame(end_time, math.ceil) if self._get_nearest_video_frame(end_time, math.ceil) < (num_frames-1) else self._get_nearest_video_frame(end_time, math.ceil)-1,
                            ]
                        ),
                        new_dict["a_timestamps"].append(
                            [
                                self._get_nearest_audio_frame(start_time, math.floor),
                                self._get_nearest_audio_frame(end_time, math.ceil) if self._get_nearest_audio_frame(end_time, math.ceil) < (a_num_frames-1) else self._get_nearest_audio_frame(end_time, math.ceil)-1,
                            ]
                        )
                formatted_data[clip_uid] = new_dict
        return formatted_data, clip_video_map

    def _get_final_dataset(self):
        """Apply filters to the dataset"""
        print("Adding filters to data...")
        for _id, info in tqdm(self.sample_query_map.items(), total=len(self.sample_query_map.keys()) ,desc=f"Final episodic nlq {self.split} dataset"):
            #number of sample option
            if self.number_of_sample is not None:
                if len(self.data) >= self.number_of_sample:
                    break

            if self.filter_vids is not None and info['video_id'] not in self.filter_vids:
                continue
            
            # modalities option
            if self.modalities is not None:
                if (Modal._Video in self.modalities):
                    if  (info['clip_path'] == None):
                        continue

                if (Modal._Audio in self.modalities): 
                    if (info['audio_path'] == None) :
                        continue

            # number of frames as options
            if self.min_frames is not None:
                if info['annotated_frame_length'] < self.min_frames:
                    continue
            if self.max_frames is not None:
                if info['annotated_frame_length'] > self.max_frames:
                    continue

            self.data.append((_id, info))

    def _load_default_audio(self, path):
        """Load default audio"""
        print("Loading default audio...")
        return torch.load(path)
    
    def _process_dataset(self, data):
        """Process the entire json"""
        print("Processing data...")
        # use cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # text features models
        tokenizer = AutoTokenizer.from_pretrained(self.parsed_args.wordEmbedding_model)
        model = AutoModel.from_pretrained(self.parsed_args.wordEmbedding_model, output_hidden_states = True ).to(device) # Whether the model returns all hidden-states.
        model.eval()         

        for clp, data_item in tqdm(data.items(), total=len(data), desc=f"process episodic nlq {self.split}"): 

            num_frames = data_item["num_frames"]
            num_audio_frames = data_item["a_num_frames"]
            zipper = zip(
                        data_item["a_timestamps"],
                        data_item["timestamps"],
                        data_item["exact_times"],
                        data_item["sentences"],
                        data_item["annotation_uids"],
                        data_item["query_idx"],
            )

            #modal paths
            clip_path = os.path.join(self.video_features_path, clp+'.pt')
            if not os.path.exists(clip_path):
                clip_path = None

            audio_path = os.path.join(self.audio_features_path, clp+'.pt')
            if not os.path.exists(audio_path):
                audio_path = None
  
            for a_timestamps, timestamp, exact_time, sentence, ann_uid, query_idx in zipper:

                s_frame = 0
                e_frame = num_frames-1
                s_audio_frame = 0
                e_audio_frame = num_audio_frames-1

                if self.split == "train": #at test give the whole clip as input
                    #negative padding
                    padding = random.randint(0,50)
                    s_frame = max(0, timestamp[0]-padding)
                    e_frame = min(num_frames-1, timestamp[1]+padding)
                    s_audio_frame = max(0, a_timestamps[0]-padding)
                    e_audio_frame = min(num_audio_frames-1, a_timestamps[1]+padding)

                #tokenizer for bert with [cls] token
                _query = sentence.strip().lower()
                input = tokenizer(_query, return_tensors='pt').to(device)
                _text_features = None
                with torch.no_grad():
                    _text_features = model(**input).last_hidden_state.to('cpu')

                words = sentence

                record =  {
                    "video_id": str(self.all_clip_video_map[clp][0]),
                    "clip_id": str(clp),
                    "clip_path": clip_path,
                    "audio_path": audio_path,
                    "words": words,
                    "query": _query,
                    "query_features": _text_features,
                    "annotation_uid": ann_uid,
                    "query_idx": query_idx,
                    "s_video_frame": timestamp[0],
                    "e_video_frame": timestamp[1],
                    "s_audio_frame": a_timestamps[0],
                    "e_audio_frame": a_timestamps[1],
                    "exact_s_time": exact_time[0],
                    "exact_e_time": exact_time[1],
                    "clip_s_time": self.all_clip_video_map[clp][3],
                    "clip_e_time": self.all_clip_video_map[clp][4],
                    "video_frame_length": e_frame - s_frame + 1,
                    "audio_frame_length": e_audio_frame - s_audio_frame + 1,
                    "video_range": ( s_frame,  e_frame),
                    "audio_range": ( s_audio_frame,  e_audio_frame),
                    "annotated_frame_length": timestamp[1] - timestamp[0]
                }

                self.sample_query_map[self.idx_sample_query] = record
                self.idx_counter += e_frame - s_frame + 1
                self.idx_sample_query += 1

def get_train_loader(dataset, batch_size):
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
    )
    return train_loader

def get_test_loader(dataset, batch_size):
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_collate_fn,
    )
    return test_loader

def train_collate_fn(batch):
    sample_id, clip_id, clip_features, audio_features, query_features, is_s, is_e, is_ans, info = zip(*batch)
    if clip_features[0] is None: #TODO
        return clip_id, clip_features, query_features, is_s, is_e, is_ans #TODO

    # TODO have to pad different clip lengths with some token - make loss fn ignore those too
    clip_id = [x for x in clip_id]
    sample_id = [x for x in sample_id]
    assert len(clip_features) == 1
    clip_features = torch.stack(clip_features)

    # default_audio = info[0]['default_audio'].repeat(clip_features[0].shape[0], 1).to(clip_features)
    default_audio = torch.zeros(clip_features[0].shape[0],info[0]['Audio Feature Size']).to(clip_features)
    audio_features = [torch.mean(x,dim=1) if x is not None else default_audio for x in audio_features]
    audio_features = torch.stack(audio_features)
    
    #get only CLS embedding for query
    query_features = [torch.cat([y[:,0,:] for y in x],dim=0) for x in query_features]
    query_features = torch.stack(query_features)

    is_s = torch.stack([torch.tensor(x) for x in is_s]).to(torch.float)
    is_e = torch.stack([torch.tensor(x) for x in is_e]).to(torch.float)
    is_ans = torch.stack([torch.tensor(x) for x in is_ans]).to(torch.float)
    # frame_length = torch.stack([torch.tensor(x) for x in frame_length])

    return sample_id, clip_id, clip_features, audio_features, query_features, is_s, is_e, is_ans, info