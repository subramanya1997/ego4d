#!/bin/sh
#SBATCH --job-name=meme
#SBATCH -o /work/shantanuagar_umass_edu/pushed/ego4d/logs/center_batched.txt
#SBATCH --time=80:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=64GB  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 2  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate memes
export TRANSFORMERS_CACHE=/work/shantanuagar_umass_edu/ego4d/meme/ego4d/cache/
cd /work/shantanuagar_umass_edu/pushed/ego4d
# python train.py -p fix_seed -w train_joint_fq >logs/train_joint_fq.out
# python train.py -p long_run -w train_joint --learning_rate 0.008 --loss_weight 0.92289625 --loss_weight2 0.972772 --num_epochs 80 --resume --load-path joint.pth >logs/train_joint.out

# python train.py -p long_run -w joint_pretrained_ --learning_rate 0.0027 --loss_weight 0.89221 --loss_weight2 0.9907 --loss_weight3 0.5 --clip_window 80 --num_epochs 80  >logs/joint_pretrained1.out
#2395780

# python train2.py -p long_run -w pre_500_best_swp --learning_rate 0.00745 --loss_weight 0.98536 --loss_weight2 0.89625 --loss_weight3 0.2599 --clip_window 80 --num_epochs 80  --resume --load-path sweepy5.pth >logs/pre_500_best_swp.out
#2400257

# python train3_cat.py -p joint -w cat_aud_450 --learning_rate 0.0027 --loss_weight 0.89221 --loss_weight2 0.9907 --loss_weight3 0.5 >logs/cat_aud_450.out
# 2949088

python train_batched.py -p test -w center_batched --learning_rate 0.0027 --num_epochs 80 >logs/center_batched.out
# 2949091