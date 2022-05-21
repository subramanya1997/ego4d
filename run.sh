#!/bin/sh
#SBATCH --job-name=encoder_decoder
#SBATCH -o test_job.%j.txt
#SBATCH --time=14-00:00:00
#SBATCH --mem=100GB  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH --gres gpu 
#SBATCH -G 1  # Number of GPUs

nvidia-smi

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate py39

cd /work/snagabhushan_umass_edu/ego4d_test2/
# # python infer.py -p fix_infer -w orig --load-path /work/shantanuagar_umass_edu/ego4d/model/nlq/meme_long_6_last.pth >sbatch/out/fix_infer.txt
# # python train.py -w orig >sbatch/out/fix_infer.txt

python main.py > sbatch/out/train_encoder_decoder_layer12.txt