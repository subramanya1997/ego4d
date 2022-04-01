#!/bin/sh
#SBATCH --job-name=meme
#SBATCH -o /home/shantanuagar_umass_edu/model_meme/ego4d/output/logs/meme_supervised_trans_w_6.txt
#SBATCH --time=10:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=64GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate sandbox
cd /home/shantanuagar_umass_edu/model_meme/ego4d
# python train.py -w trans_weighted >sbatch/out/meme_supervised_trans_w_2.txt
python train.py -w trans_weighted_0.2 --loss_weight 0.2 >sbatch/out/meme_supervised_trans_w_6.txt
#### trans- 2348826, 2349005, 2349033, 2349035, 2349038, 2349042