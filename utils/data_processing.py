#import libs
import json
import os
from tqdm import tqdm
import math

#import pytorch
import torch
from torch.utils.data import Dataset, DataLoader

#import transformer
from transformers import BertTokenizer

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
            number_of_sample (int): None reads all values, or number of samples
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
        
        assert (
            type(raw_data) == dict
        ), "Annotation file format {} not supported.".format(type(raw_data))
        
        raw_data, clip_video_map = self._reformat_data(raw_data, self.split == "test")
        self.all_clip_video_map.update(clip_video_map)
        self._process_frame_embeddings(raw_data, features_path)
        print(f"#{self.split}: {self.idx_counter}")
        
        if save_or_load or update:
            self.save_data(save_or_load_path)
    

    def __len__(self):
        return self.idx_counter

    def __getitem__(self, idx):
        clip_path = self.data[idx]['clip_path']
        clip_feature = torch.load(clip_path)
        clip_id = self.data[idx]['clip_id']
        is_s = self.data[idx]['is_s_frame']
        is_e = self.data[idx]['is_e_frame']
        is_ans = self.data[idx]['is_within_range']
        return clip_id, clip_feature, is_s, is_e, is_ans
    
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

    def _reformat_data(self, split_data, test_split=False):
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

                        if not test_split:
                            start_time = float(datum["clip_start_sec"])
                            end_time = float(datum["clip_end_sec"])
                        else:
                            # Random placeholders for test set.
                            start_time = 0.
                            end_time = 0.

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
            
        self.data = []
        for clp, data_item in tqdm(data.items(), total=len(data), desc=f"process episodic nlq {self.split}"):
            
            if self.number_of_sample is not None:
                if self.number_of_sample <= self.idx_counter:
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
                s_frame = max(0, timestamp[0]-5)
                e_frame = min(num_frames, timestamp[1]+5)
                words = sentence
                for frame_num in range(s_frame, e_frame+1):
                    record = {
                        "sample_id": self.idx_counter,
                        "clip_id": str(clp),
                        "clip_path": clip_path,
                        # "clip_feature": clip_feature,
                        "words": words,
                        "query": sentence.strip().lower(),
                        "word_emb": tokenizer(sentence.strip().lower()),
                        "annotation_uid": ann_uid,
                        "query_idx": query_idx,
                        "s_frame": timestamp[0],
                        "e_frame": timestamp[1],
                        "exact_s_time": exact_time[0],
                        "exact_e_time": exact_time[1],
                        "frame_num": frame_num,
                        "is_s_frame": True if frame_num == timestamp[0] else False,
                        "is_e_frame": True if frame_num == timestamp[1] else False,
                        "is_within_range": True if frame_num >= timestamp[0] and frame_num <= timestamp[1] else False,
                    }
                    self.data.append(record)
                    self.idx_counter += 1

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
