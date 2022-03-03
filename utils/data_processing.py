#import libs
import json
import os
from tqdm import tqdm
import math

#import pytorch
import torch
from torch.utils.data import Dataset, DataLoader

#import transformer
from transformers import BertTokenizer, BertModel

#import custom functions
from utils.data_utils import get_nearest_frame, load_pickle, save_pickle
from config import Config

class Ego4d_NLQ(Dataset):
    def __init__(self, annotations_path, features_path, split="train", wordEmbedding="bert", number_of_sample=None, save_or_load=False, update=False, save_or_load_path="./scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/train.pkl"):
        """Class for reading and visualizing annotations.
        Args:
            annotations_path (str): location of annotation file
            features_path (str): location of clip features file
            split (str): train, val or test
            wordEmbedding (str): bert (model for text embeddings)
            number_of_sample (int): None reads all values, or number of annotations
            save_or_load (bool): save to pickle (Not able to save the whole file)
            update (bool): to update save pickel file
            save_or_load_path (str): path to load or save
        """
        if save_or_load and not update:
            if os.path.exists(save_or_load_path): #load
                saved_data = load_pickle(save_or_load_path)
                self.version = saved_data['version']
                self.description = saved_data['description']
                self.all_clip_video_map = saved_data['all_clip_video_map']
                self.split = saved_data['split']
                self.wordEmbedding =  saved_data['wordEmbedding']
                self.idx_counter = saved_data['idx_counter']
                self.data = saved_data['data']
                return

        raw_data = self._load_json(annotations_path)
        self.version = raw_data['version']
        self.description = raw_data['description']
        self.all_clip_video_map = {}
        self.split = split
        self.wordEmbedding = wordEmbedding
        self.number_of_sample = number_of_sample
        
        self.idx_counter = 0
        self.sample_query_map = {}
        self.idx_sample_query = 0
        
        assert (
            type(raw_data) == dict
        ), "Annotation file format {} not supported.".format(type(raw_data))
        
        raw_data, clip_video_map = self._reformat_data(raw_data)
        self.all_clip_video_map.update(clip_video_map)
        self._process_frame_embeddings(raw_data, features_path)

        print(f"#{self.split}: {self.idx_counter}")
        
        if save_or_load or update:
            self.save_data(save_or_load_path)

        # get feature sizes
        if self.idx_counter != 0:
            _, clip_feature, query_features, _, _, _, _ = self[0]
            self.video_feature_size = clip_feature.shape[-1]
            self.query_feature_size = query_features.shape[-1]
        else:
            print('No Data Loaded!')

    def __len__(self):
        if self.split=="train":
            return self.idx_counter
        else:
            return len(self.sample_query_map)

    def __getitem__(self, idx):
        if self.split == "train":
            clip_path = self.data[idx]['clip_path']
            clip_feature = torch.load(clip_path)
            frame_num = self.data[idx]['frame_num']
            clip_feature = clip_feature[frame_num,:]
            clip_id = self.data[idx]['clip_id']
            query_features = self.data[idx]['query_features']
            is_s = self.data[idx]['is_s_frame']
            is_e = self.data[idx]['is_e_frame']
            is_ans = self.data[idx]['is_within_range']
            frame_length = self.data[idx]['frame_length']
            return clip_id, clip_feature, query_features, is_s, is_e, is_ans, frame_length
        
        return self.get_test_query(idx)
    
    def get_test_query(self,idx):
        sample_query = self.sample_query_map[idx]
        s_idx, e_idx = sample_query["range"]
        data = self.data[s_idx:e_idx]
        clip_path = data[0]['clip_path']
        clip_id = data[0]['clip_id']
        clip_features = torch.load(clip_path)

        assert (clip_features.shape[0] == len(data)) and len(list(set([x['clip_id'] for x in data]))) == 1

        query_features = [item['query_features'] for item in data]
        is_s = [item['is_s_frame'] for item in data]
        is_e = [item['is_e_frame'] for item in data]
        is_ans = [item['is_within_range'] for item in data]
        frame_length = [item['frame_length'] for item in data]
        return clip_id, clip_features, query_features, is_s, is_e, is_ans, frame_length

    def get_query_sample(self, idx):
        '''
        '''
        sample_query = self.sample_query_map[idx]
        s_idx, e_idx = sample_query["range"]
        items = []
        for _idx in range(s_idx, e_idx):
            items.append(self.__getitem__(_idx))
        return items
    
    def save_data(self, path):
        saved_data = {}
        saved_data['version'] = self.version
        saved_data['description'] = self.description
        saved_data['all_clip_video_map'] = self.all_clip_video_map
        saved_data['split'] = self.split
        saved_data['wordEmbedding'] = self.wordEmbedding
        saved_data['idx_counter'] =  self.idx_counter
        saved_data['data'] = self.data

        if not os.path.exists(path): #save create folder
            pass
        save_pickle(saved_data, path)

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
    
    def _get_nearest_frame(self, time, floor_or_ceil=None):
        """Obtain the nearest frame for a given time, video fps, and feature window."""
        return floor_or_ceil(int(time * Config.VIDEO_FPS / Config.WINDOW_SIZE))

    def _process_question(self, question):
        """Process the question to make it canonical."""
        return question.strip(" ").strip("?").lower() + "?"

    def _reformat_data(self, split_data):
        """Convert the format from JSON files.
        fps, num_frames, timestamps, sentences, exact_times,
        annotation_uids, query_idx.
        """
        formatted_data = {}
        clip_video_map = {}
        for video_datum in split_data["videos"]:
            for clip_datum in video_datum["clips"]:
                clip_uid = clip_datum["clip_uid"]
                clip_video_map[clip_uid] = (
                    video_datum["video_uid"],
                    clip_datum["video_start_sec"],
                    clip_datum["video_end_sec"],
                )
                clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
                num_frames = self._get_nearest_frame(clip_duration, math.ceil)
                new_dict = {
                    "fps": Config.VIDEO_FPS / Config.WINDOW_SIZE,
                    "num_frames": num_frames,
                    "timestamps": [],
                    "exact_times": [],
                    "sentences": [],
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
                        new_dict["annotation_uids"].append(ann_datum["annotation_uid"])
                        new_dict["query_idx"].append(index)
                        new_dict["exact_times"].append([start_time, end_time]),
                        new_dict["timestamps"].append(
                            [
                                get_nearest_frame(start_time, math.floor),
                                get_nearest_frame(end_time, math.ceil),
                            ]
                        )
                formatted_data[clip_uid] = new_dict
        return formatted_data, clip_video_map
    
    def _process_frame_embeddings(self, data, features_path):
        tokenizer = None
        if self.wordEmbedding == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True ) # Whether the model returns all hidden-states.
            model.eval()         
            
        self.data = []
        for clp, data_item in tqdm(data.items(), total=len(data), desc=f"process episodic nlq {self.split}"):
            
            if self.number_of_sample is not None:
                if self.number_of_sample <= self.idx_sample_query:
                    break

            num_frames = data_item["num_frames"]
            zipper = zip(
                        data_item["timestamps"],
                        data_item["exact_times"],
                        data_item["sentences"],
                        data_item["annotation_uids"],
                        data_item["query_idx"],
            )
            clip_path = os.path.join(features_path, clp+'.pt')
            # clip_feature = None 
            # if os.path.exists(features_path):
            #     clip_feature = torch.load(clip_path)

            for timestamp, exact_time, sentence, ann_uid, query_idx in zipper:

                if self.number_of_sample is not None:
                    if self.number_of_sample <= self.idx_sample_query:
                        break

                s_frame = 0
                e_frame = num_frames
                if "test" != self.split: #at test give the whole clip as input
                    s_frame = max(0, timestamp[0]-5)
                    e_frame = min(num_frames-1, timestamp[1]+5)

                _query_idx_range = [self.idx_counter]

                #tokenizer for bert with [cls] token
                _query = sentence.strip().lower()
                input = tokenizer(_query, return_tensors='pt')
                _word_features = None
                with torch.no_grad():
                    _word_features = model(**input).last_hidden_state

                words = sentence
                for frame_num in range(s_frame, e_frame+1):
                    record = {
                        "sample_id": self.idx_counter,
                        "clip_id": str(clp),
                        "clip_path": clip_path,
                        # "clip_feature": clip_feature,
                        "words": words,
                        "query": _query,
                        "query_features": _word_features,
                        "annotation_uid": ann_uid,
                        "query_idx": query_idx,
                        "sample_query_idx": self.idx_sample_query,
                        "s_frame": timestamp[0],
                        "e_frame": timestamp[1],
                        "exact_s_time": exact_time[0],
                        "exact_e_time": exact_time[1],
                        "frame_num": frame_num,
                        "frame_length": e_frame - s_frame,
                        "is_s_frame": True if frame_num == timestamp[0] else False,
                        "is_e_frame": True if frame_num == timestamp[1] else False,
                        "is_within_range": True if frame_num >= timestamp[0] and frame_num <= timestamp[1] else False,
                    }
                    self.data.append(record)
                    self.idx_counter += 1
                
                self.sample_query_map[self.idx_sample_query] = {
                    "clip_id": str(clp),
                    "annotation_uid": ann_uid,
                    "query_idx": query_idx,
                    "range" : _query_idx_range.append( self.idx_counter )   
                }
                self.idx_sample_query += 1


def get_train_loader(dataset, batch_size):
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=train_collate_fn,
    )
    return train_loader

def get_test_loader(dataset, batch_size):
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=test_collate_fn,
    )
    return test_loader

def train_collate_fn(batch):
    clip_id, clip_features, query_features, is_s, is_e, is_ans, frame_length = zip(*batch)
    clip_features = torch.stack(clip_features)
    query_features = torch.stack(query_features)

    print(is_s)
    is_s = torch.stack(is_s)
    return batch