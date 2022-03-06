import os
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse

def get_all_audio_files(args):
    file_paths = defaultdict(str)
    for root, dirs, files in os.walk(args['input_audio_directory']):
        for file in files:
            if file.endswith(".pt"):
                file_paths[file.split('.')[0]] = os.path.join(root, file)
    video_files_path = defaultdict(str)
    for root, dirs, files in os.walk(args['input_video_directory']):
        for file in files:
            if file.endswith(".pt"):
                video_files_path[file.split('.')[0]] = os.path.join(root, file)
    print("Number of audio files found in ",args['input_audio_directory']," is ", len(file_paths.keys()))
    print("Number of audio files found in ",args['input_audio_directory']," is ", len(video_files_path.keys()))
    return file_paths, video_files_path

def checkNumberOfDatapoints(paths, video_files_path, args):
    tempT = 0
    tempF = 0
    for _filename, _path in tqdm(paths.items()):
        if _filename in video_files_path:
            _audio_datapoints = torch.load(_path).shape[0] // 49
            _video_datapoints = torch.load(os.path.join(args['input_video_directory'], _filename+'.pt')).shape[0]
            print(_audio_datapoints, _video_datapoints)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_audio_directory", required=True, help="Path to audio"
    )
    parser.add_argument(
        "--input_video_directory", required=True, help="path to video",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    paths, video_files_path = get_all_audio_files(parsed_args)
    checkNumberOfDatapoints(paths, video_files_path, parsed_args)
