#!/bin/bash
# Slurm script for DIET inference on Brown CS Hydra cluster
#
#SBATCH --partition=gpus
#SBATCH --exclude=gpu[1601-1605],gpu[1701-1708],gpu1801,gpu[1802,1905-1906]
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --job-name=DIET_INFERENCE
#SBATCH --output=/home/jambsdor/projects/DIET_Tuning/logs/inference_%A_%a.out
#SBATCH --error=/home/jambsdor/projects/DIET_Tuning/logs/inference_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jakob_ambsdorf@brown.edu

# Navigate to project directory
cd /home/jambsdor/projects/DIET_Tuning

# Activate virtual environment
source .venv/bin/activate

PROJECT_DIR=/home/jambsdor/projects/DIET_Tuning
DATA_DIR=/data/people/jambsdor/diet_data

# Set environment variables
export WANDB_DATA_DIR=$DATA_DIR/wandb

# Get the wandb ID for this array task
CSV_FILE="$1"
PYTHON_ARGS="$2"

# Read the specific line for this array index (SLURM_ARRAY_TASK_ID)
WANDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CSV_FILE" | tr -d '\r\n' | grep -v '^#' | grep -v '^[[:space:]]*$')

if [ -z "$WANDB_ID" ]; then
    echo "Error: Could not read wandb ID for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing wandb ID: $WANDB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Starting inference at $(date)"

# Run the inference
srun python -u test.py \
    --wandb-id "$WANDB_ID" \
    --data-root $DATA_DIR \
    --wandb-dir $WANDB_DATA_DIR \
    --checkpoint-dir $DATA_DIR/checkpoints \
    $PYTHON_ARGS

echo "Completed inference at $(date)"
