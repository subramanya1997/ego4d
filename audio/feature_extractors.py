import argparse

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import os
from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np


class Audio_clip(Dataset):
    def __init__(self, audio_path, processor, window = 8000, device='cpu' ):
        self.audio_data, self.sample_rate = sf.read(audio_path)
        self.processor = processor
        self._window = window
        self.device = device
        #print(self.audio_data.shape, " Shape Before Process")
        self._process_audio()
        #print(self.audio_shape, " Shape after Process")
        
        #print(self[self.__len__() -1], self.__len__())
        
    def __len__(self):
        return self.audio_shape[0]
    
    def __getitem__(self, idx):
        return self.audio_data[idx]
        
    def _process_audio(self):
        _extra = self.audio_data.shape[0] % self.sample_rate
        #print(f'Appending { self.sample_rate - _extra } zeros')
        self.audio_data = np.append(self.audio_data, np.zeros( self.sample_rate - _extra, dtype = self.audio_data.dtype))
        self.audio_data = sliding_window_view(self.audio_data, self.sample_rate )[:: self._window , :]
        self.audio_data = self.audio_data
        self.audio_shape = self.audio_data.shape
        self.audio_data = [self.processor(x, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)['input_values'][0] for x in self.audio_data]

def extract_features(paths, features_path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    processor = Wav2Vec2Processor.from_pretrained(args['model'])
    model = Wav2Vec2Model.from_pretrained(args['model'], output_hidden_states = True).to(device)
    model.eval()
    os.makedirs(args["audio_feature_save_path"], exist_ok=True)
    feature_sizes = {}
    for _filename, _path in tqdm(paths.items()):
        
        if _filename in features_path:
            _audio_data, _sample_rate = sf.read(_path)
            _temp_fd = torch.load(features_path[_filename])
            if (((_temp_fd.shape[0] // 49) * (_sample_rate/2)) - _sample_rate) <= _audio_data.shape[0]:
                feature_sizes[_filename] = _temp_fd.shape[1]
                continue
        
        dataset = Audio_clip(_path, processor)
        train_loader = DataLoader(
                dataset=dataset,
                batch_size=128,
                shuffle=False
            )
        audio_features = []
    
        for i in train_loader:
            with torch.no_grad():
                temp = model(i.to(device)).last_hidden_state
                audio_features.extend(torch.flatten(temp.to('cpu'), end_dim=1)) 
        audio_features = torch.stack(audio_features)

        feature_save_path = os.path.join(args["audio_feature_save_path"], _filename+".pt")    
        torch.save(audio_features.to('cpu'), feature_save_path)
        
        feature_sizes[_filename] = audio_features.shape[1]

    save_path = os.path.join(args["audio_feature_save_path"], "feature_shapes.json")
    with open(save_path, "w") as file_id:
        json.dump(feature_sizes, file_id)

def get_all_audio_files(args):
    file_paths = defaultdict(str)
    for root, dirs, files in os.walk(args['input_audio_directory']):
        for file in files:
            if file.endswith(".wav"):
                file_paths[file.split('.')[0]] = os.path.join(root, file)
    
    audio_feature_paths = defaultdict(str)
    for root, dirs, files in os.walk(args['audio_feature_save_path']):
        for file in files:
            if file.endswith(".pt"):
                audio_feature_paths[file.split('.')[0]] = os.path.join(root, file)

    print("Number of audio files found in ",args['input_audio_directory']," is ", len(file_paths.keys()))
    return file_paths, audio_feature_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="model",
    )
    parser.add_argument(
        "--input_audio_directory", required=True, help="Path to audio"
    )
    parser.add_argument(
        "--audio_feature_save_path", required=True, help="Path to save clip audio features",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    paths, features_path = get_all_audio_files(parsed_args)
    extract_features(paths, features_path, parsed_args)