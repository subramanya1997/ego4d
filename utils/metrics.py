import numpy as np

from transformers import pipeline

def decode_candidate_clips(start, end, topk = 1, max_len = None):
    qa_pipeline = pipeline("question-answering")
    qa_pipeline.decode()
    mask = np.ones(start.shape[0])
    max_len = start.shape[0] if max_len is None else max_len
    s, e, scores = qa_pipeline.decode(start, end,topk=topk,max_answer_len=max_len, undesired_tokens=mask)
    return s, e, scores


