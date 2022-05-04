#! /usr/bin/env python
"""
Prepare Ego4d episodic memory NLQ for model training.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import collections
import csv
import json
import math
import yaml
import os

import torch
import tqdm


CANONICAL_VIDEO_FPS = 30.0
FEATURE_WINDOW_SIZE = 16.0
FEATURES_PER_SEC = CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE


def get_nearest_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, video fps, and feature window."""
    return floor_or_ceil(int(time * CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE))


def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"


def reformat_data(split_data):
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
            num_frames = get_nearest_frame(clip_duration, math.ceil)
            new_dict = {
                "fps": FEATURES_PER_SEC,
                "num_frames": num_frames,
                "timestamps": [],
                "exact_times": [],
                "sentences": [],
                "annotation_uids": [],
                "query_idx": [],
            }

            for ann_datum in clip_datum["annotations"]:
                for index, datum in enumerate(ann_datum["language_queries"]):
                    start_time = float(datum["clip_start_sec"])
                    end_time = float(datum["clip_end_sec"])
                    if "query" not in datum or not datum["query"]:
                        continue
                    new_dict["sentences"].append(process_question(datum["query"]))
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

def reformat_data_mq(split_data):
        """Convert the format from JSON files.
        fps, num_frames, timestamps, sentences, exact_times,
        annotation_uids, query_idx.
        """
        print("Reforming data...")
        formatted_data = {}
        clip_video_map = {}
        for video_datum in split_data["videos"]:
            for clip_datum in video_datum["clips"]:
                clip_uid = clip_datum["clip_uid"]
                clip_video_map[clip_uid] = (
                    video_datum["video_uid"],
                    clip_datum["video_start_sec"],
                    clip_datum["video_end_sec"],
                    clip_datum["clip_start_sec"],
                    clip_datum["clip_end_sec"],
                )
                clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
                num_frames = get_nearest_frame(clip_duration, math.ceil)
                a_num_frames = num_frames
                new_dict = {
                    "fps": FEATURES_PER_SEC,
                    "num_frames": num_frames,
                    "a_num_frames": a_num_frames,
                    "timestamps": [],
                    "a_timestamps": [],
                    "exact_times": [],
                    "query_labels": [],
                    "query_timestamps": [],
                    "query_frames": [],
                    "annotation_uids": [],
                    "query_idx": [],
                }

                for ann_datum in clip_datum["annotations"]:
                    if "labels" not in ann_datum:
                        continue
                    for index, datum in enumerate(ann_datum["labels"]):
                        start_time = float(datum["start_time"])
                        end_time = float(datum["end_time"])

                        new_dict["query_labels"].append(datum["label"])
                        new_dict["query_timestamps"].append([datum["video_start_time"], datum["video_end_time"]])
                        new_dict["query_frames"].append([datum["video_start_frame"], datum["video_end_frame"]])
                        new_dict["annotation_uids"].append(ann_datum["annotator_uid"])
                        new_dict["query_idx"].append(index)
                        new_dict["exact_times"].append([start_time, end_time]),
                        new_dict["timestamps"].append(
                            [
                                get_nearest_frame(start_time, math.floor),
                                get_nearest_frame(end_time, math.ceil),
                            ]
                        ),
                        new_dict["a_timestamps"].append(
                            [
                                get_nearest_frame(start_time, math.floor),
                                get_nearest_frame(end_time, math.ceil),
                            ]
                        )
                formatted_data[clip_uid] = new_dict
        return formatted_data, clip_video_map

def reformat_data_vq(split_data):
    """Convert the format from JSON files.
    fps, num_frames, timestamps, sentences, exact_times,
    annotation_uids, query_idx.
    """
    print("Reforming data...")
    formatted_data = {}
    clip_video_map = {}
    for video_datum in split_data["videos"]:
        for clip_datum in video_datum["clips"]:
            clip_uid = clip_datum["clip_uid"]
            clip_video_map[clip_uid] = (
                video_datum["video_uid"],
                clip_datum["video_start_sec"],
                clip_datum["video_end_sec"],
                clip_datum["clip_start_sec"],
                clip_datum["clip_end_sec"],
            )
            clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
            num_frames = get_nearest_frame(clip_duration, math.ceil)
            a_num_frames = num_frames
            new_dict = {
                "fps": FEATURES_PER_SEC,
                "num_frames": num_frames,
                "a_num_frames": a_num_frames,
                "query_video_frames": [],
                "query_frames": [],
                "query_response_tracks": [],
                "query_object_titles": [],
                "query_visual_crops": [],
                "query_key": [],
                "query_idx": [],
            }

            for ann_datum in clip_datum["annotations"]:
                if "query_sets" not in ann_datum:
                    continue
                for index, (k, datum) in enumerate(ann_datum["query_sets"].items()):
                    if "is_valid" in datum and datum["is_valid"] is not True:
                        continue

                    new_dict["query_video_frames"].append(datum["query_video_frame"])
                    new_dict["query_frames"].append(datum["query_frame"])
                    new_dict["query_response_tracks"].append(datum["response_track"])
                    new_dict["query_object_titles"].append(datum["object_title"])
                    new_dict["query_visual_crops"].append(datum["visual_crop"])
                    new_dict["query_idx"].append(index)
                    new_dict["query_key"].append(k)
                    
            formatted_data[clip_uid] = new_dict
    return formatted_data, clip_video_map

def convert_ego4d_dataset(args):
    """Convert the Ego4D dataset for VSLNet."""
    # Reformat the splits to train vslnet.
    all_clip_video_map = {}
    for split in ("train", "val", "test"):
        read_path = args[f"input_{split}_split"]
        if read_path is None or not os.path.exists(read_path):
            print("{} does not exist".format(read_path))
            continue
        print(f"Reading [{split}]: {read_path}")
        with open(read_path, "r") as file_id:
            raw_data = json.load(file_id)
        data_split, clip_video_map = reformat_data(raw_data)
        all_clip_video_map.update(clip_video_map)
        num_instances = sum(len(ii["sentences"]) for ii in data_split.values())
        print(f"# {split}: {num_instances}")

        save_path = os.path.join(args["output_save_path"], f"{split}.json")
        print(f"Writing [{split}]: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(data_split, file_id)

    # Extract visual features based on the all_clip_video_map.
    feature_sizes = {}
    os.makedirs(args["clip_feature_save_path"], exist_ok=True)
    progress_bar = tqdm.tqdm(all_clip_video_map.items(), desc="Extracting features")
    for clip_uid, (video_uid, start_sec, end_sec) in progress_bar:
        feature_path = os.path.join(args["video_feature_read_path"], f"{video_uid}.pt")
        feature = torch.load(feature_path)

        # Get the lower frame (start_sec) and upper frame (end_sec) for the clip.
        clip_start = get_nearest_frame(start_sec, math.floor)
        clip_end = get_nearest_frame(end_sec, math.ceil)
        clip_feature = feature[clip_start : clip_end + 1]
        feature_sizes[clip_uid] = clip_feature.shape[0]
        feature_save_path = os.path.join(
            args["clip_feature_save_path"], f"{clip_uid}.pt"
        )
        torch.save(clip_feature, feature_save_path,_use_new_zipfile_serialization=False)

    save_path = os.path.join(args["clip_feature_save_path"], "feature_shapes.json")
    with open(save_path, "w") as file_id:
        json.dump(feature_sizes, file_id)

def convert_ego4d_dataset_mq(args):
    """Convert the Ego4D dataset for VSLNet."""
    # Reformat the splits to train vslnet.
    all_clip_video_map = {}
    for split in ("train", "val", "test"):
        read_path = args[f"input_{split}_split"]
        if read_path is None or not os.path.exists(read_path):
            print("{} does not exist".format(read_path))
            continue
        print(f"Reading [{split}]: {read_path}")
        with open(read_path, "r") as file_id:
            raw_data = json.load(file_id)
        if args["task"] == "mq":
            data_split, clip_video_map = reformat_data_mq(raw_data)
        else :
            data_split, clip_video_map = reformat_data_vq(raw_data)
        all_clip_video_map.update(clip_video_map)
        t = args["task"]
        save_path = os.path.join(args["output_save_path"], f"{t}_{split}.json")
        print(f"Writing [{split}]: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(data_split, file_id)

    # Extract visual features based on the all_clip_video_map.
    feature_sizes = {}
    os.makedirs(args["clip_feature_save_path"], exist_ok=True)
    progress_bar = tqdm.tqdm(all_clip_video_map.items(), desc="Extracting features")
    for clip_uid, (video_uid, start_sec, end_sec, _, _) in progress_bar:
        feature_path = os.path.join(args["video_feature_read_path"], f"{video_uid}.pt")
        if feature_path is None or not os.path.exists(feature_path):
            print("{} does not exist".format(feature_path))
            continue
        feature = torch.load(feature_path)

        # Get the lower frame (start_sec) and upper frame (end_sec) for the clip.
        clip_start = get_nearest_frame(start_sec, math.floor)
        clip_end = get_nearest_frame(end_sec, math.ceil)
        clip_feature = feature[clip_start : clip_end + 1]
        feature_sizes[clip_uid] = clip_feature.shape[0]
        feature_save_path = os.path.join(
            args["clip_feature_save_path"], f"{clip_uid}.pt"
        )
        torch.save(clip_feature, feature_save_path,_use_new_zipfile_serialization=False)

    save_path = os.path.join(args["clip_feature_save_path"], "feature_shapes.json")
    with open(save_path, "w") as file_id:
        json.dump(feature_sizes, file_id)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", required=True
    )

    # Override arguments if specified in command line
    parser.add_argument(
        "--input_train_split", help="Path to Ego4d train split"
    )
    parser.add_argument(
        "--input_val_split", help="Path to Ego4d val split"
    )
    parser.add_argument(
        "--input_test_split", help="Path to Ego4d test split"
    )
    parser.add_argument(
        "--output_save_path", help="Path to save the output jsons"
    )
    parser.add_argument(
        "--video_feature_read_path", help="Path to read video features"
    )
    parser.add_argument(
        "--clip_feature_save_path", help="Path to save clip video features",
    )
    parser.add_argument(
        "--task", help="nlq/mq/vq", default="nlq"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    
    # Read config yamls file
    config_file = parsed_args.pop("config_file")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if key not in parsed_args or parsed_args[key] is None:
            parsed_args[key] = value

    print("Parsed Arguments are - ", parse_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args["task"] == "mq":
        print("MQ task")
        convert_ego4d_dataset_mq(parsed_args)
    elif parsed_args["task"] == "vq":
        print("VQ task")
        convert_ego4d_dataset_mq(parsed_args)
    else:
        convert_ego4d_dataset(parsed_args)
