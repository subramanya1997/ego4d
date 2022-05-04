import os
import json
from collections import defaultdict
from numpy import random

def get_task_video_ids(file_ = "data/annotations/nlq_train.json"):
    with open(file_) as f:
        data = json.load(f)

    video_ids = [x['video_uid'] for x in data['videos']]
    return video_ids

def get_filtered_video_metadata(video_ids, main_file = "/work/sreeragiyer_umass_edu/ego4d_data/ego4d.json"):
    with open(main_file) as f:
        main_data = json.load(f)
    em_vids = {x['video_uid']:x for x in main_data['videos'] if x['video_uid'] in video_ids}
    print(len(em_vids), len(video_ids))

    return em_vids

def get_metadata2uids(em_vids, test_file = None):
    scenarios,video_source,device,has_redacted_regions = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
    if test_file is not None and os.path.exists(test_file):
        with open(test_file) as f:
            test_data = json.load(f)
            test_vid_ids = [x['video_uid'] for x in test_data['videos']]
            # filter out the videos that are not in the test set
            em_vids = {x:y for x,y in em_vids.items() if x not in test_vid_ids}
            print("#Vids in Test Set:",len(test_vid_ids))
            print("#Vids After Removing Test Vids:",len(em_vids))

    for id,z in em_vids.items():
        scenes = z['scenarios']
        for s in scenes:
            scenarios[s].append(id)
        video_source[z['video_source']].append(id)
        device[z['device']].append(id)
        has_redacted_regions[z['has_redacted_regions']].append(id)
    return [scenarios,video_source,device,has_redacted_regions]

def get_audio_video_ids(file_= "data/annotations/nlq_train.json", folder="/scratch/snagabhushan_umass_edu/dataset/v1/audio/", test_file = None):
    with open(file_) as f:
        data = json.load(f)
    clips_with_audio = [x.split(".wav")[0] for x in os.listdir(folder) if x.endswith(".wav")]
    vids_w_audio = [x['video_uid'] for x in data['videos'] if len(x['clips'])>0 and x['clips'][0]['clip_uid'] in clips_with_audio]
    print("#Vids w Audio:",len(vids_w_audio))

    if test_file is not None and os.path.exists(test_file):
        with open(test_file) as f:
            test_data = json.load(f)
            test_vid_ids = [x['video_uid'] for x in test_data['videos']]
            # filter out the videos that are not in the test set
            vids_w_audio = [x for x in vids_w_audio if x not in test_vid_ids]
            print("#Vids w Audio After Removing Test Vids:",len(vids_w_audio))

    return vids_w_audio

def get_stats(file_,folder):
    with open(file_) as f:
        data = json.load(f)

    # Total Annotatations
    clips = [y for x in data['videos'] for y in x['clips']]
    annotations = [y for x in clips for y in x['annotations']]
    print("#Videos in All Data:",len(data['videos']))
    print("#Clips in All Data:",len(clips))
    print("#Annotations in All Data:",len(annotations))

    clips_with_audio = [x.split(".wav")[0] for x in os.listdir(folder) if x.endswith(".wav")]
    all_clips_w_audio = [y for x in data['videos'] for y in x['clips'] if y['clip_uid'] in clips_with_audio]
    vids_w_audio = [x['video_uid'] for x in data['videos'] if len(x['clips'])>0 and x['clips'][0]['clip_uid'] in clips_with_audio]
    print("#Vids w Audio:",len(vids_w_audio))
    print("#Clips w Audio:",len(all_clips_w_audio))
    all_annotations_w_audio = [y for x in all_clips_w_audio for y in x['annotations']]
    print("#Annotations w Audio:",len(all_annotations_w_audio))

def save_sample_data(dict_list, main_file, save_file_ = "./data/annotations/sample/sample_nlq_train.json", sample_portion=0.10,audio_vids=None):
    sampled_vids = []
    print("Sampling:")
    for dic in dict_list:
        for k,v in dic.items():
            # select 10% of the videos in each category and add to sampled vids
            new_samples = random.choice(v,int(len(v)*sample_portion),replace=False)
            sampled_vids.extend(new_samples)
            # print(f"#{k}: {len(new_samples)}")
    sampled_vids = list(set(sampled_vids))

    # check for audio vids
    if audio_vids is not None:
        sampled_audio_vids = [x for x in sampled_vids if x in audio_vids]
        if len(sampled_audio_vids) < int(len(audio_vids)*sample_portion): #less than 10 percent of the sampled vids have audio
            n_audio_samples = int(sample_portion * len(audio_vids)) - len(sampled_audio_vids)

            new_samples = random.choice([x for x in audio_vids if x not in sampled_audio_vids],int(n_audio_samples),replace=False)
            sampled_vids.extend(new_samples)
            print("#Audio Samples:",len(sampled_audio_vids)+len(new_samples))
            sampled_vids = list(set(sampled_vids))

    with open(main_file) as f:
        main_data = json.load(f)
    

    main_data['videos'] = [x for x in main_data['videos'] if x['video_uid'] in sampled_vids]
    
    print("#Total Sampled Vids:",len(sampled_vids))
    # save the sampled vids to a file
    with open(save_file_,"w") as f:
        json.dump(main_data,f,indent=4)

if __name__ == "__main__":
    file_ = "/work/sreeragiyer_umass_edu/ego4d_data/v1/annotations/vq_train.json"
    audio_path = "/work/snagabhushan_umass_edu/dataset/v1/audio/"
    save_file_ = "/work/sreeragiyer_umass_edu/ego4d_data/v1/samples/sample_vq_train.json"
    test_file = None
    # test_file = "./data/annotations/sample/sample_nlq_test.json"
    sample_percent = 0.1
    get_stats(file_,audio_path)
    video_ids = get_task_video_ids(file_)
    em_vids = get_filtered_video_metadata(video_ids)
    dict_list = get_metadata2uids(em_vids, test_file = test_file)
    audio_vids = get_audio_video_ids(file_, audio_path, test_file = test_file)
    save_sample_data(dict_list, file_, audio_vids = audio_vids, save_file_ = save_file_, sample_portion=sample_percent)
    get_stats(save_file_,audio_path)
