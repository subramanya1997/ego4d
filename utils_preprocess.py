#import libs
import json
import os
from collections import defaultdict
import moviepy.editor as mp
from tqdm import tqdm
import pickle
import csv 
import soundfile as sf

from config import Config

def readJsonFile(path="./dataset/tmp/ego4d.json"):
    with open(path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
        return jsonObject
    
    
def extractAudioFromClips(video_path="./dataset/tmp/v1/clips/", audio_path="./dataset/tmp/v1/audio/"):
    no_audio_found = []
    audio_found = []
    details = ['clip_id', 'path']
    file_paths = defaultdict(str)
    for root, dirs, files in os.walk(video_path):
        for file in files:
            if file.endswith(".mp4"):
                file_paths[file.split('.')[0]] = os.path.join(root, file)
                
    for _filename, _path in tqdm(file_paths.items()):
        _audio_path = os.path.join(audio_path, _filename + ".wav")
        
        _audio = mp.AudioFileClip(_path)

        if os.path.exists(_audio_path):
            print(_audio_path, " has audio path" )
            temp_audio, temp_sr = sf.read(_audio_path)
            if int(_audio.duration) == (temp_audio //temp_sr):
                audio_found.append( [_filename, _audio_path] )
                continue
        print(_filename, _audio_path)
        
        try:
            _audio.get_frame(0)

            _audio.write_audiofile(_audio_path, fps=16e3, codec="pcm_s32le", ffmpeg_params=["-ac", "1"])  
            audio_found.append( [_filename, _audio_path] )
        except:
            print(f"Audio not found for clip id {_filename}!")
            no_audio_found.append( [_filename, _path] )

    with open('./dataset/tmp/v1/audio/audio_not_found.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(details) 
        write.writerows(no_audio_found) 

    with open('./dataset/tmp/v1/audio/manifest.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(details) 
        write.writerows(audio_found) 

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