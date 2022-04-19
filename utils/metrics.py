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
        # score/=(end-start+1) # Normalize
        segments.append((score, start, end))

    segments = sorted(segments, key=lambda x: x[0], reverse=True)
    starts = [x[1] for x in segments[:topk]]
    ends = [x[2] for x in segments[:topk]]
    scores = [x[0] for x in segments[:topk]]
    return starts, ends, scores

def multi_infer(pred, topk):
    segment_scores = pred[:,0,-1]
    
    segment_scores = pred[:,0,-1]
    best_segments = np.argsort(segment_scores)[::-1][:topk]

    scored_segments = []
    for i in best_segments:
        segment = pred[i:i+1,:,-1]
        s_, e_, scores_ = get_best_segment(segment, topk)
        proposals = [(scores_[i], s_[i], e_[i]) for i in range(len(scores_))]
        scored_segments+=proposals

    segments = sorted(scored_segments, key=lambda x: x[0], reverse=True)
    starts = [x[1] for x in segments[:topk]]
    ends = [x[2] for x in segments[:topk]]
    scores = [x[0] for x in segments[:topk]]
    return starts, ends, scores

def get_best_scoring_segment(preds, topk=5):
    '''
    preds are logits
    '''
    assert preds.shape[0]==1
    # find longest sequence with max log sum score
    log_pred = np.log(preds[0])
    zeros = log_pred[:,0]
    ones = preds[0,:,1]

    outer = np.matmul(np.expand_dims(ones[None], -1), np.expand_dims(ones[None], 1))
    ones = np.log(outer+1e-10)[0]

    zeros_f = np.cumsum(zeros)
    zeros_f[1:] = zeros_f[:-1]
    zeros_f[0] = 0
    zeros_f = zeros_f[:,None]

    zeros_b = np.cumsum(zeros[::-1])[::-1]
    zeros_b[:-1] = zeros_b[1:]
    zeros_b[-1] = 0
    zeros_b = zeros_b[:,None]

    scores = ones+zeros_f+zeros_b
    candidates = np.tril(np.triu(scores), scores.shape[1])
    candidates[candidates == 0] = -np.inf

    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    starts, ends = np.unravel_index(idx_sort, candidates.shape)
    scores = candidates[starts, ends]
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