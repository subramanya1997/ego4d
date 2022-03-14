import numpy as np

def decode_candidate_clips(qa_pipeline, start, end, topk = 5, max_len = None):
    mask = np.ones(start.shape[0])
    max_len = start.shape[0] if max_len is None else max_len
    s, e, scores = qa_pipeline.decode(start, end,topk=topk,max_answer_len=max_len, undesired_tokens=mask)
    return s, e, scores

def get_best_segment(preds, topk=5):
    # find longest sequence with max log sum score
    mask = preds>0.5
    log_pred = np.log(preds)
    segments = []
    i=0
    start = -1
    end = -1
    while i<log_pred.shape[1]:
        if not mask[0,i]:
            i+=1
            continue
        score = 0
        start = i
        while i<log_pred.shape[1] and mask[0,i]:
            score += log_pred[0, i]
            i+=1
        end = i-1
        segments.append((score, start, end))

    segments = sorted(segments, key=lambda x: x[0], reverse=True)
    starts = [x[1] for x in segments[:topk]]
    ends = [x[2] for x in segments[:topk]]
    scores = [x[0] for x in segments[:topk]]
    return starts, ends, scores

# def eval_test(
#     result_save_path=None,
#     gt_json_path=None,
#     epoch=None,
#     global_step=None,
# ):
#     # Save predictions if path is provided.
#     if result_save_path:
#         with open(result_save_path, "w") as file_id:
#             json.dump(predictions, file_id)

#     # Evaluate if ground truth JSON file is provided.
#     if gt_json_path:
#         with open(gt_json_path) as file_id:
#             ground_truth = json.load(file_id)
#         thresholds = [0.3, 0.5, 0.01]
#         topK = [1, 3, 5]
#         results, mIoU = ego4d_eval.evaluate_nlq_performance(
#             predictions, ground_truth, thresholds, topK
#         )
#         title = f"Epoch {epoch}, Step {global_step}"
#         score_str = ego4d_eval.display_results(
#             results, mIoU, thresholds, topK, title=title
#         )
#     else:
#         results = None
#         mIoU = None
#         score_str = None
#     return results, mIoU, score_str