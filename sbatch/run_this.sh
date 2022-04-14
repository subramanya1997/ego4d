#!/bin/sh
#SBATCH --job-name=meme
#SBATCH -o /scratch/shantanuagar_umass_edu/ego4d/meme/ego4d/output/logs/long_train_input.txt
#SBATCH --time=20:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=64GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate sandbox
cd /scratch/shantanuagar_umass_edu/ego4d/meme/ego4d
python train.py -w long_train_input >sbatch/out/long_train_input.txt
#### 2355368