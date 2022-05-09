from collections import defaultdict
import torch
import os
from tqdm import tqdm

paths = defaultdict(str)
for root, dirs, files in os.walk("/work/snagabhushan_umass_edu/dataset/v1/clip_features/"):
    for file in files:
        if file.endswith(".pt"):
            paths[file.split('.')[0]] = os.path.join(root, file)
avg_length = 0
count = 0
for i, path in tqdm(paths.items()):
    load_pt = torch.load(path)
    avg_length += load_pt.shape[0]
    if load_pt.shape[0] < 1000:
        count += 1

print(avg_length/len(paths), count)