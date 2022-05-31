#!/bin/sh
#SBATCH --job-name=encoder_decoder
#SBATCH -o test_job.%j.txt
#SBATCH --time=14-00:00:00
#SBATCH --mem=150GB  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH --gres gpu 
#SBATCH -G 1  # Number of GPUs

nvidia-smi

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate py39

cd /work/snagabhushan_umass_edu/ego4d_test2/
# # python infer.py -p fix_infer -w orig --load-path /work/shantanuagar_umass_edu/ego4d/model/nlq/meme_long_6_last.pth >sbatch/out/fix_infer.txt
# # python train.py -w orig >sbatch/out/fix_infer.txt

# python main.py --model_name encoderdecoder --load_model ./checkpoints/encoderdecoder_nlq_official_v1_official_128_bert/model/encoderdecoder_10351.t7 > sbatch/out/train_encoder_decoder_layer12.txt

# python main.py --model_name appendQuery_256 > sbatch/out/appendQuery.txt --load_model ./checkpoints/appendQuerylr1e4_nlq_official_v1_official_128_bert/model/appendQuerylr1e4_37640.t7
python main.py --model_name appendQuery__128 --max_pos_len 128 --dim 128 > sbatch/out/appendQuery__128.txt