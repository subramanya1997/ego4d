#import libs
import json
import os
from pydoc import cli
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

from collections import defaultdict

class Ego4d_NLQ(Dataset):
    def __init__(self, annotations_path, features_path=None, split="train", wordEmbedding="bert", number_of_sample=None, numer_of_frames=500, save_or_load=False, update=False, save_or_load_path="./scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/train.pkl"):
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
                self.video_feature_size = saved_data['video_feature_size']
                self.query_feature_size = saved_data['query_feature_size']
                self.sample_query_map = saved_data['sample_query_map']
                self.numer_of_frames = saved_data['numer_of_frames']
                self.idx_sample_query = saved_data['idx_sample_query']
                # for i in range(len(self.sample_query_map.keys())):
                #     _, clip_feature, query_features, _, _, _, _, = self[i]
                
                print(f"#{self.split} frames: {self.idx_counter}")
                print(f"#{self.split} clips: {self.idx_sample_query}")

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
        self.numer_of_frames = numer_of_frames

        self.ids_remove = defaultdict(int)
        
        assert (
            type(raw_data) == dict
        ), "Annotation file format {} not supported.".format(type(raw_data))
        
        raw_data, clip_video_map = self._reformat_data(raw_data)
        self.raw_data_formatted = raw_data
        self.all_clip_video_map.update(clip_video_map)
        if(features_path != None):
            self._process_frame_embeddings(raw_data, features_path)

        print(f"#{self.split} frames: {self.idx_counter}")
        print(f"#{self.split} clips: {self.idx_sample_query}")

        # get feature sizes
        if self.idx_counter != 0:
            #if self.split == 'train':
            if False:
                _, clip_feature, query_features, _, _, _, _, _ = self[0]
                self.video_feature_size = clip_feature.shape[-1]
                self.query_feature_size = query_features.shape[-1]
            else:
                _, clip_feature, query_features, _, _, _, _ = self[0]
                self.video_feature_size = clip_feature[0].shape[-1]
                self.query_feature_size = query_features[0].shape[-1]
        else:
            print('No Data Loaded!')

        for i in range(len(self.sample_query_map.keys())):
            _, clip_feature, query_features, _, _, _, _ = self[i]

        print(self.ids_remove)

        if save_or_load or update:
            self.save_data(save_or_load_path)

    def __len__(self):
        #if self.split=="train":
        if False:
            return self.idx_counter
        else:
            return len(self.sample_query_map)

    def __getitem__(self, idx):
        
        #if self.split == "train":
        if False:
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
            sample_query_idx = self.data[idx]['sample_query_idx']
            return clip_id, clip_feature, query_features, is_s, is_e, is_ans, frame_length, sample_query_idx
        
        return self.get_test_query(idx)
    
    def get_test_query(self,idx):
        sample_query = self.sample_query_map[idx]
        s_d_idx, e_d_idx = sample_query["range"]
        data = self.data[ s_d_idx : e_d_idx ]
        s_idx, e_idx = sample_query["clip_range"]
        clip_path = data[0]['clip_path']
        clip_id = data[0]['clip_id']
        clip_features = torch.load(clip_path)
        
        # print(clip_features.shape, s_idx, e_idx)
        
        clip_features = clip_features[ s_idx : e_idx , : ]

        if clip_id == '8d3f5b12-ac2c-4315-80ed-bb827aa91bd4':
            return None, None, None, None, None, None, None
        
        if (clip_features.shape[0] != len(data)) and len(list(set([x['clip_id'] for x in data]))) != 1:
            self.ids_remove[clip_id] += 1
            print('Wrong')

        query_features = [item['query_features'] for item in data]
        is_s = [item['is_s_frame'] for item in data]
        is_e = [item['is_e_frame'] for item in data]
        is_ans = [item['is_within_range'] for item in data]
        frame_length = [item['frame_length'] for item in data]
        return clip_id, clip_features, query_features, is_s, is_e, is_ans, frame_length


    def getfromidx(self, idx):
        sample_map = self.sample_query_map[idx]
        frame = self.data[sample_map["range"][0]]
        values = {
            "annotation_uid": sample_map["annotation_uid"],
            "clip_id": sample_map["clip_id"],
            "query_idx": sample_map["query_idx"],
            "query": frame["query"],
            "s_time": frame["exact_s_time"],
            "e_time": frame["exact_e_time"],

        }
        return values

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
        saved_data['video_feature_size'] = self.video_feature_size 
        saved_data['query_feature_size'] = self.query_feature_size
        saved_data['sample_query_map'] = self.sample_query_map
        saved_data['numer_of_frames'] = self.numer_of_frames
        saved_data['idx_sample_query'] = self.idx_sample_query

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
                                get_nearest_frame(start_time, math.floor),
                                get_nearest_frame(end_time, math.ceil),
                            ]
                        )
                formatted_data[clip_uid] = new_dict
        return formatted_data, clip_video_map
    
    def _process_frame_embeddings(self, data, features_path):
        tokenizer = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.wordEmbedding == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True ).to(device) # Whether the model returns all hidden-states.
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
                e_frame = num_frames-1

                if self.split == "train": #at test give the whole clip as input
                    s_frame = max(0, timestamp[0]-5)
                    e_frame = min(num_frames-1, timestamp[1]+5)

                clip_features = torch.load(clip_path)
                clip_features = clip_features[ s_frame : e_frame , : ]
                
                if (clip_features.shape[0] != (e_frame - s_frame)):
                    continue

                if (timestamp[1] - timestamp[0]) < 5:
                    continue

                if self.numer_of_frames is not None:
                    if self.numer_of_frames < (e_frame - s_frame):
                        continue

                _s_index_query =  self.idx_counter

                #tokenizer for bert with [cls] token
                _query = sentence.strip().lower()
                input = tokenizer(_query, return_tensors='pt').to(device)
                _word_features = None
                with torch.no_grad():
                    _word_features = model(**input).last_hidden_state.to('cpu')

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
                    "range" : ( _s_index_query, self.idx_counter ),
                    "clip_range": ( s_frame,  e_frame+1)  
                }
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
    clip_id, clip_features, query_features, is_s, is_e, is_ans, frame_length = zip(*batch)
    if clip_features[0] is None: #TODO
        return clip_id, clip_features, query_features, is_s, is_e, is_ans #TODO

    clip_id = [x for x in clip_id]
    assert len(clip_features) == 1
    clip_features = [x.squeeze(0) for x in clip_features]
    clip_features = clip_features[0]

    #get only CLS embedding for query
    query_features = [x[:,0,:] for x in query_features[0]]
    query_features = torch.cat(query_features, dim=0)
    is_s = torch.stack([torch.tensor(x) for x in is_s]).to(torch.float)[0]
    is_e = torch.stack([torch.tensor(x) for x in is_e]).to(torch.float)[0]
    is_ans = torch.stack([torch.tensor(x) for x in is_ans]).to(torch.float)[0]
    frame_length = torch.stack([torch.tensor(x) for x in frame_length])[0]

    return clip_id, clip_features, query_features, is_s, is_e, is_ans