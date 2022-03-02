from utils.data_processing import Ego4d_NLQ

nql = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_train.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="train", wordEmbedding="bert", number_of_sample=None, save_or_load=False, update=True, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/train_all.pkl")
print(nql.data[0]['query_features'].shape)