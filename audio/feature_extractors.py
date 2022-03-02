import argparse

import soundfile as sf
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import os
from tqdm import tqdm
import json
from collections import defaultdict

def extract_features(paths, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    processor = Wav2Vec2Processor.from_pretrained(args['model'])
    model = Wav2Vec2Model.from_pretrained(args['model'], output_hidden_states = True).to(device)
    model.eval()
    os.makedirs(args["audio_feature_save_path"], exist_ok=True)
    feature_sizes = {}
    for _filename, _path in tqdm(paths.items()):
        audio_input, sample_rate = sf.read(_path)
        inputs = processor(audio_input[:16000], sampling_rate=sample_rate, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        audio_features = outputs.last_hidden_state.to('cpu')

        feature_save_path = os.path.join(args["audio_feature_save_path"], _filename+".pt")
        torch.save(audio_features, feature_save_path)

        feature_sizes[_filename] = audio_features.shape[0]

    save_path = os.path.join(args["audio_feature_save_path"], "feature_shapes.json")
    with open(save_path, "w") as file_id:
        json.dump(feature_sizes, file_id)
        

def get_all_audio_files(args):
    file_paths = defaultdict(str)
    for root, dirs, files in os.walk(args['input_audio_directory']):
        for file in files:
            if file.endswith(".wav"):
                file_paths[file.split('.')[0]] = os.path.join(root, file)
    print("No of audio files found in ",args['input_audio_directory']," is ", len(file_paths.keys()))
    return file_paths


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

    paths = get_all_audio_files(parsed_args)
    extract_features(paths, parsed_args)