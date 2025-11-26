module load python3/3.11.9

# cd into project directory
cd /zhome/6a/3/198677/projects/DIET_Tuning

# Activate conda environment
source /dtu/p1/jakambs/diet_env/bin/activate

export WANDB_DATA_DIR=/dtu/p1/jakambs/diet/wandb

export CUDA_VISIBLE_DEVICES=0

python test.py \
    --wandb-id p8vrfnrg \
    --data-root /dtu/p1/jakambs/diet/data \
    --wandb-dir /dtu/p1/jakambs/diet/wandb \
    --checkpoint-dir /dtu/p1/jakambs/diet/checkpoints \
