#import libs
import json
import os
from re import L
from tqdm import tqdm
import math
import yaml
import argparse
import enum

#import pytorch
import torch
from torch.utils.data import Dataset, DataLoader

#import transformer
from transformers import BertTokenizer, BertModel

#import custom functions
from utils.data_utils import load_pickle, save_pickle

from collections import defaultdict

class Modal(enum.Enum):
    _Video = 1
    _Audio = 2
    _Transcript = 3
    _IMU = 4
    _3D = 5

class Ego4d_MQ(Dataset):
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
                self.audio_feature_size = saved_data['audio_feature_size']
                self.query_feature_size = saved_data['query_feature_size']
                #models and options
                self.wordEmbedding_model =  self.parsed_args.wordEmbedding_model
                self.max_frames = self.parsed_args.max_frames if self.parsed_args.max_frames != 'None' else None
                self.min_frames = self.parsed_args.min_frames if self.parsed_args.min_frames != 'None' else None
                self.number_of_sample = self.parsed_args.number_of_sample if self.parsed_args.number_of_sample != 'None' else None
                self.video_features_path = self.parsed_args.video_features_path
                self.audio_features_path = self.parsed_args.audio_features_path
                self.modalities = set(modalities) if modalities is not None else None
                self.filter_vids = set(filter_vids) if filter_vids is not None else None

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
        
        assert (
            self.video_features_path is not None
        ), "No Video Features Path Found - Please update dataset_config.yaml"

        assert (
            self.audio_features_path is not None
        ), "No Audio Features Path Found - Please update dataset_config.yaml"
        
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

        # get feature sizes
        if self.idx_counter != 0:
            """ sample_id, clip_id, clip_features, audio_features, query_features, is_s, is_e, is_ans, frame_length """
            _, _, clip_feature, audio_features, query_features, _, _, _, _ = self[0]
            if clip_feature is not None:
                self.video_feature_size = clip_feature[0].shape[-1]
            if audio_features is not None:
                self.audio_feature_size = audio_features[0][0].shape[-1]
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
        
        clip_path = sample_query['clip_path']
        clip_id = sample_query['clip_id']

        audio_path = sample_query['audio_path']

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

        query_label_features = sample_query['query_label_features']
        
        is_s = [ (sample_query['s_video_frame'] == i) for i in range(s_v_idx, e_v_idx)]
        is_e = [ (sample_query['e_video_frame'] == i) for i in range(s_v_idx, e_v_idx)]
        is_ans = [ (sample_query['s_video_frame'] <= i and sample_query['e_video_frame'] >= i) for i in range(s_v_idx, e_v_idx)]
        frame_length = sample_query['video_frame_length']#[ sample_query['video_frame_length'] for i in range(s_v_idx, e_v_idx)]

        info = {"Video Feature Size": self.video_feature_size,
                "Audio Feature Size": self.audio_feature_size,
                "Query Feature Size": self.query_feature_size,
                "Frame length": frame_length}
        
        return sample_id, clip_id, clip_features, audio_features, query_label_features, is_s, is_e, is_ans, info

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
                    "query_labels": [],
                    "query_timestamps": [],
                    "query_frames": [],
                    "annotation_uids": [],
                    "query_idx": [],
                }

                for ann_datum in clip_datum["annotations"]:
                    if "labels" not in ann_datum:
                        continue
                    for index, datum in enumerate(ann_datum["labels"]):
                        start_time = float(datum["start_time"])
                        end_time = float(datum["end_time"])

                        new_dict["query_labels"].append(datum["label"])
                        new_dict["query_timestamps"].append([datum["video_start_time"], datum["video_end_time"]])
                        new_dict["query_frames"].append([datum["video_start_frame"], datum["video_end_frame"]])
                        new_dict["annotation_uids"].append(ann_datum["annotator_uid"])
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
        for _id, info in tqdm(self.sample_query_map.items(), total=len(self.sample_query_map.keys()) ,desc=f"Final episodic mq {self.split} dataset"):
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

    
    def _process_dataset(self, data):
        """Process the entire json"""
        print("Processing data...")
        # use cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # text features models
        tokenizer = BertTokenizer.from_pretrained(self.parsed_args.wordEmbedding_model)
        model = BertModel.from_pretrained(self.parsed_args.wordEmbedding_model, output_hidden_states = True ).to(device) # Whether the model returns all hidden-states.
        model.eval()         

        for clp, data_item in tqdm(data.items(), total=len(data), desc=f"process episodic mq {self.split}"): 
            num_frames = data_item["num_frames"]
            num_audio_frames = data_item["a_num_frames"]
            zipper = zip(
                        data_item["a_timestamps"],
                        data_item["timestamps"],
                        data_item["exact_times"],
                        data_item["query_timestamps"],
                        data_item["query_frames"],
                        data_item["query_labels"],
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
  
            for a_timestamps, timestamp, exact_time, query_timestamp, query_frame, query_label, ann_uid, query_idx in zipper:

                s_frame = 0
                e_frame = num_frames-1
                s_audio_frame = 0
                e_audio_frame = num_audio_frames-1

                if self.split == "train": #at test give the whole clip as input
                    s_frame = max(0, timestamp[0]-5)
                    e_frame = min(num_frames-1, timestamp[1]+5)
                    s_audio_frame = max(0, a_timestamps[0]-5)
                    e_audio_frame = min(num_audio_frames-1, a_timestamps[1]+5)

                #tokenizer for bert with [cls] token
                input = tokenizer(query_label, return_tensors='pt').to(device)
                _text_features = None
                with torch.no_grad():
                    _text_features = model(**input).last_hidden_state.to('cpu')

                record =  {
                    "video_id": str(self.all_clip_video_map[clp][0]),
                    "clip_id": str(clp),
                    "clip_path": clip_path,
                    "audio_path": audio_path,
                    "query_start_time": query_timestamp[0],
                    "query_end_time": query_timestamp[1],
                    "query_start_frame": query_frame[0],
                    "query_end_frame": query_frame[1],
                    "query_label": query_label,
                    "query_label_features": _text_features,
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
