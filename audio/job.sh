#!/bin/bash
#python feature_extractors.py --model facebook/wav2vec2-large-960h-lv60-self --input_audio_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_extracted_of_video/ --audio_feature_save_path /scratch/snagabhushan_umass_edu/dataset/v1/audio_features_from_video/
#python validates.py --input_audio_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_extracted_of_video/ --input_video_directory /scratch/shantanuagar_umass_edu/ego4d/saved_clip_features

python feature_extractors_for_clips.py --annotations_path /scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_test.json --input_audio_feature_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_features_from_video/ --audio_feature_save_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_features/
python feature_extractors_for_clips.py --annotations_path /scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_train.json --input_audio_feature_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_features_from_video/ --audio_feature_save_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_features/
python feature_extractors_for_clips.py --annotations_path /scratch/shantanuagar_umass_edu/ego4d/nlq/sample/sample_nlq_val.json --input_audio_feature_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_features_from_video/ --audio_feature_save_directory /scratch/snagabhushan_umass_edu/dataset/v1/audio_features/