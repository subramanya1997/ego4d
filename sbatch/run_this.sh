#!/bin/sh
#SBATCH --job-name=meme
#SBATCH -o /work/shantanuagar_umass_edu/pushed/ego4d/logs/train_joint_fq.txt
#SBATCH --time=20:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=64GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate memes

cd /work/shantanuagar_umass_edu/pushed/ego4d
python train.py -p fix_seed -w train_joint_fq >logs/train_joint_fq.out
#2381752