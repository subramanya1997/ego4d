import os
import json
import copy

from utils.data_processing import Ego4d_NLQ
import utils.evaluate_ego4d_nlq as ego4d_eval
from config import Config

def evaluate_predicted_records(predicted_records, epoch, gt_path, nlq_data):
    predictions = parse_predictions(predicted_records, nlq_data)
    results, mIoU, score_str, metric_dict = evaluate_w_gt(gt_path,predictions,epoch)
    # print(score_str)
    return results, mIoU, score_str, metric_dict

def evaluate_nlq_predictions(nlq_val, nlq_test, record_path, gt_path, gt_path_test):
    all_records = os.listdir(record_path)
    val_records = sorted([x for x in all_records if ".json" in x and "test" not in x])[-2:]
    test_records = sorted([x for x in all_records if ".json" in x and "test" in x])[-2:]

    print("Evaluating for VAL")
    for name in val_records:
        with open(os.path.join(record_path, name), "r") as f:
            records = json.load(f)
        epoch = int(name.split(".json")[0].split("_")[-1])
        results, mIoU, score_str, metric_dict = evaluate_predicted_records(records, epoch, gt_path, nlq_val)
        print(score_str)
        print(metric_dict)

    print("Evaluating for TEST")
    for name in test_records:
        with open(os.path.join(record_path, name), "r") as f:
            records = json.load(f)
        epoch = int(name.split(".json")[0].split("_")[-1])
        #something
        results, mIoU, score_str, _ = evaluate_predicted_records(records, epoch, gt_path_test, nlq_val)
        print(score_str)

def parse_predictions(records, nlq_val):
    predictions = []

    for record in records:
        clip_uid = record['clip_id']
        clip_uid = clip_uid.split("'")[1]
        sample_id = record['sample_id']
        values = nlq_val.getfromidx(sample_id)

        timewindow_predictions = []
        for s,e in zip(record['start'], record['end']):
            start_time, end_time = index_to_time(s, e)
            timewindow_predictions.append([float(start_time), float(end_time)])
        
        new_datum = {
            "clip_uid": clip_uid,
            "annotation_uid": values["annotation_uid"],
            "query_idx": int(values["query_idx"]),
            "predicted_times": copy.deepcopy(timewindow_predictions),
            "query": values["query"],  # for sanity check purposes.
            "recorded_time": [values["s_time"], values["e_time"]],
        }
        predictions.append(new_datum)

    # Save predictions if path is provided.
    return predictions

# Evaluate if ground truth JSON file is provided.
def evaluate_w_gt(gt_json_path,predictions,epoch):
    if gt_json_path:
        with open(gt_json_path) as file_id:
            ground_truth = json.load(file_id)
        thresholds = [0.3, 0.5, 0.01]
        topK = [1, 3, 5]
        results, mIoU = ego4d_eval.evaluate_nlq_performance(
            predictions, ground_truth, thresholds, topK
        )
        title = f"Epoch {epoch}"
        score_str = ego4d_eval.display_results(
            results, mIoU, thresholds, topK, title=title
        )
        metric_dict = ego4d_eval.results_dict(
            results, mIoU, thresholds, topK, title=title
        )
    else:
        results = None
        mIoU = None
        score_str = None
        metric_dict = None
    return results, mIoU, score_str, metric_dict

def index_to_time(start_index, end_index):
    start_time = start_index * Config.WINDOW_SIZE / Config.VIDEO_FPS
    end_time = end_index * Config.WINDOW_SIZE / Config.VIDEO_FPS
    return start_time, end_time


# if __name__ == "__main__":
#     nlq_val = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_val.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="val", wordEmbedding="bert", number_of_sample=25, numer_of_frames=750, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/val_25.pkl")
#     nlq_test = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_test.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="test", wordEmbedding="bert", number_of_sample=25, numer_of_frames=750, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/test_25.pkl")
#     gt_path = '/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_val.json'
#     record_path = "output/records/"
    
#     evaluate_nlq_predictions(nlq_val, nlq_test, record_path, gt_path)