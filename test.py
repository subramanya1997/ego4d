from data_processing import Ego4d_NLQ

nql = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_train.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="train", wordEmbedding="bert", number_of_sample=1000, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/train.pkl")
print(len(nql.data))