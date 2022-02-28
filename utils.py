#import libs
import json
import os
from collections import defaultdict
import moviepy.editor as mp

def readJsonFile(path="./dataset/tmp/ego4d.json"):
    with open(path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
        return jsonObject
    
    
def extractAudioFromClips(video_path="./dataset/tmp/v1/clips/", audio_path="./dataset/tmp/v1/audio/"):
    file_paths = defaultdict(str)
    for root, dirs, files in os.walk("./dataset/tmp/v1/clips/"):
        for file in files:
            if file.endswith(".mp4"):
                file_paths[file.split('.')[0]] = os.path.join(root, file)
                
    for _filename, _path in file_paths.items():
        _clip = mp.VideoFileClip(_path)
        _clip.audio.write_audiofile(os.path.join(audio_path, _filename, ".mp3"))
        