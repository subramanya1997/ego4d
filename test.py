from utils.data_processing import Ego4d_NLQ, Modal

nql = Ego4d_NLQ('/scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_train.json', modalities=None, split="train", save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/final_train.pkl")

nql = Ego4d_NLQ('/scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_val.json', modalities=None, split="val", save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/final_val.pkl")

nql = Ego4d_NLQ('/scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_test.json', modalities=None, split="test", save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/final_test.pkl")