#import libs
import json
import os
from collections import defaultdict
import moviepy.editor as mp
from tqdm import tqdm
import pickle

from config import Config

def readJsonFile(path="./dataset/tmp/ego4d.json"):
    with open(path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
        return jsonObject
    
    
def extractAudioFromClips(video_path="./dataset/tmp/v1/clips/", audio_path="./dataset/tmp/v1/audio/"):
    file_paths = defaultdict(str)
    for root, dirs, files in os.walk(video_path):
        for file in files:
            if file.endswith(".mp4"):
                file_paths[file.split('.')[0]] = os.path.join(root, file)
                
    for _filename, _path in tqdm(file_paths.items()):
        print(_filename, _path)
        _audio = mp.AudioFileClip(_path)
        _audio.write_audiofile(os.path.join(audio_path, _filename + ".wav"), fps=16e3, codec="pcm_s32le", ffmpeg_params=["-ac", "1"])  

def get_nearest_audio_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, audo fps, and feature window."""
    return floor_or_ceil(int(time * Config.AUDIO_FPS / Config.AUDIO_WINDOW))

def get_nearest_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, video fps, and feature window."""
    return floor_or_ceil(int(time * Config.VIDEO_FPS / Config.WINDOW_SIZE))

def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
        return data

def save_pickle(data, filename):
    with open(filename, mode="wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)