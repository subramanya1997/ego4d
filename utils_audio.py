import multiprocessing  
import os
import argparse
from tqdm import tqdm
import csv
import moviepy.editor as mp
import pandas as pd
import soundfile as sf


def validate(args):
    manf = pd.read_csv(os.path.join(args['save_audio_path'], 'manifest.csv'))  

    def worker( rows, worker_id):
        for index, row in tqdm(rows.iterrows()):
            #_video_path = os.path.join(args['video_path'], row.clip_id + '.mp4')
            _audio_path = row.path
            _audio_a = mp.AudioFileClip(_audio_path)
            #if _audio_v.to_soundarray().shape != _audio_a.to_soundarray().shape:
            #    print(f"--- Audio clip found with clip id {row.clip_id} is faulty! ---")
                
    
    jobs = []
    chunk_size = len(manf) // args['num_workers'] + 1                      
    for worker_id in range(args['num_workers']):
        end = (worker_id + 1) * chunk_size
        if end >= len(manf):
            end = len(manf)
        allotment = manf[ worker_id * chunk_size : end]
        process = multiprocessing.Process(target=worker, args=( allotment, worker_id))
        jobs.append(process)
        process.start()
        

def extractor(args):
    
    file_paths = []
    for root, _, files in os.walk(args['video_path']):
        for file in files:
            if file.endswith(".mp4"):
                file_paths.append( (file.split('.')[0], os.path.join(root, file)) )
    
    print(f"Total Number of clip files are { len(file_paths) }")

    def worker( in_paths, worker_id, output_q):
        no_audio_found = []
        audio_found = []
        
        for _filename, _path in tqdm(in_paths):
            _audio_path = os.path.join(args['save_audio_path'], _filename + ".wav")

            _audio = mp.AudioFileClip(_path)

            if os.path.exists(_audio_path):
                print(_audio_path, " has audio path" )
                try:
                    temp_audio, temp_sr = sf.read(_audio_path)
                    if int(_audio.duration) == (temp_audio.size //temp_sr):
                        audio_found.append( [_filename, _audio_path] )
                        continue
                except:
                    print(f"Need to rewrite this {_filename}!")
            #print(_filename, _audio_path)
            try:
                _audio.get_frame(0)

                _audio.write_audiofile(_audio_path, fps=16e3, codec="pcm_s32le", ffmpeg_params=["-ac", "1"])  
                audio_found.append( [_filename, _audio_path] )
            except:
                print(f"Audio not found for clip id {_filename}!")
                no_audio_found.append( [_filename, _path] )

        _data = { "audio_found": audio_found, "audio_not_found": no_audio_found }
        output_q.put( {worker_id:  _data } )
    
    output_q = multiprocessing.Queue()
    jobs = []
    chunk_size = len(file_paths) // args['num_workers'] + 1                      
    for worker_id in range(args['num_workers']):
        end = (worker_id + 1) * chunk_size
        if end >= len(file_paths):
            end = len(file_paths)
        allotment = file_paths[ worker_id * chunk_size : (worker_id + 1) * chunk_size]
        print(len(allotment), chunk_size, worker_id * chunk_size, end )
        process = multiprocessing.Process(target=worker, args=( allotment, worker_id, output_q))
        jobs.append(process)
        process.start()
    #print(jobs)

    collated_results = {}
    for _ in jobs:
        collated_results.update(output_q.get())
    for job in jobs:
        job.join()

    _audio_found = []
    _audio_not_found = []
    sorted_worker_id = sorted(collated_results.keys())
    for worker_id in sorted_worker_id:
        _audio_found.extend(collated_results[worker_id]['audio_found'])
        _audio_not_found.extend(collated_results[worker_id]['audio_not_found'])

    details = ['clip_id', 'path']
    with open(os.path.join(args['save_audio_path'], 'audio_not_found.csv'), 'w') as f: 
        write = csv.writer(f) 
        write.writerow(details) 
        write.writerows(_audio_not_found) 
    with open(os.path.join(args['save_audio_path'], 'manifest.csv'), 'w') as f:
        write = csv.writer(f) 
        write.writerow(details) 
        write.writerows(_audio_found) 
                                
if __name__ == "__main__":
                                    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_path", required=True, help="clip folder", type=str
    )
    parser.add_argument(
        "--save_audio_path", required=True, help="save audio folder", type=str
    )
    parser.add_argument(
        "--num_workers", required=True, help="Number of workers", default=10, type=int
    )
    parser.add_argument(
        "--validate", required=False, help="Number of workers", default=False, type=bool
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    if parsed_args['validate']:
        validate(parsed_args)
    else:
        extractor(parsed_args)
