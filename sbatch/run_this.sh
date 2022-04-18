#!/bin/sh
#SBATCH --job-name=meme
#SBATCH -o /work/shantanuagar_umass_edu/ego4d/meme/ego4d/output/logs/train_orig.txt
#SBATCH --time=10:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=64GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH --gres gpu 
#SBATCH -G 1  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate memes

cd /work/shantanuagar_umass_edu/ego4d/meme/ego4d
# python infer.py -p fix_infer -w orig --load-path /work/shantanuagar_umass_edu/ego4d/model/nlq/meme_long_6_last.pth >sbatch/out/fix_infer.txt
# python train.py -w orig >sbatch/out/fix_infer.txt

python train.py -p fix_seed -w train_orig >sbatch/out/train_orig.txt
#2379264, 2379340


# python train2.py -p fix_seed -w best_sweep_1_full --clip_window 450 --learning_rate 0.036 --loss_weight 0.39 --loss_weight2 0.78 >sbatch/out/clip450_1_full.txt
#2379257, 2379259, 2379260

