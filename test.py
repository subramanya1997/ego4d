from utils.data_processing import Ego4d_NLQ

nql = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_train.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="train", wordEmbedding="bert", number_of_sample=75, numer_of_frames=750, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/train_50.pkl")
print(nql.data[0]['query_features'].shape)

nql = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_val.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="val", wordEmbedding="bert", number_of_sample=25, numer_of_frames=750, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/val_25.pkl")
print(nql.data[0]['query_features'].shape)

nql = Ego4d_NLQ('/scratch/snagabhushan_umass_edu/dataset/v1/annotations/nlq_test.json', '/scratch/shantanuagar_umass_edu/ego4d/saved_clip_features/', split="test", wordEmbedding="bert", number_of_sample=25, numer_of_frames=750, save_or_load=True, update=False, save_or_load_path="/scratch/snagabhushan_umass_edu/dataset/v1/save/nlq/test_25.pkl")
print(nql.data[0]['query_features'].shape)