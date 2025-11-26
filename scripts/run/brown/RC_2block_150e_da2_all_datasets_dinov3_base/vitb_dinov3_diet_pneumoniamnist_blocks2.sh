#!/bin/bash
# Slurm script for DIET block training on Brown CS Hydra cluster
#
# This script trains DINOv3-base on PneumoniaMNIST with 2 transformer blocks
#
#SBATCH --partition=gpus
#SBATCH --exclude=gpu[1601-1605],gpu[1701-1708],gpu1801,gpu[1802,1905-1906]
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=20G
#SBATCH --job-name=RC_2B_150E_BASE_PNEUM_V3
#SBATCH --output=/home/jambsdor/projects/DIET_Tuning/logs/rc_2b_150e_base_pneum_v3_%j.out
#SBATCH --error=/home/jambsdor/projects/DIET_Tuning/logs/rc_2b_150e_base_pneum_v3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jakob_ambsdorf@brown.edu


# Navigate to project directory
cd /home/jambsdor/projects/DIET_Tuning

# Activate virtual environment
source .venv/bin/activate

PROJECT_DIR=/home/jambsdor/projects/DIET_Tuning
DATA_DIR=/data/people/jambsdor/diet_data

# Set environment variables
srun nvidia-smi
export WANDB_DATA_DIR=$DATA_DIR/wandb

# Run the training script
srun python -u main.py \
    --wandb-prefix "RC_2block_150e_da2_dinov3_base" \
    --backbone dinov3 \
    --model-size base \
    --label-smoothing 0.2 \
    --da-strength 2 \
    --diet-head-only-epochs 0.05 \
    --dataset pneumoniamnist \
    --num-epochs 150 \
    --num-trained-blocks 2 \
    --data-root $DATA_DIR/data \
    --wandb-dir $DATA_DIR/wandb \
    --checkpoint-dir $DATA_DIR/checkpoints \
    $EXTRA_PYTHON_ARGS
