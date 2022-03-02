# MEME - MoE for Multimodal Egocentric Videos
experimenting with egocentric

## Preparation

### Dataset 

Download the dataset from the [official webpage][ego4d_page] and place them
in `data/` folder. Specify the paths in `config.yaml`

Run the preprocessing script using:

```bash
python utils/prepare_ego4d_dataset.py -c config.yaml
```

This creates JSON files in `data/dataset/nlq_official_v1` that can be used for training and evaluating the VSLNet baseline model.


### Video features

Download the official video features released from [official webpage][ego4d_page] and place them in `data/features/nlq_official_v1/video_features/` folder.
