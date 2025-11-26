#!/bin/bash

# Script to submit inference jobs for a list of wandb IDs using Slurm job arrays
# Usage: ./testall.sh <csv_file> [additional_python_args...]

# Check if CSV file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <csv_file> [additional_python_args...]"
    echo "Example: $0 scripts/run/inference_paths/dinov2_small_medical.csv"
    echo "Example: $0 scripts/run/inference_paths/dinov2_small_medical.csv --eval-on-test"
    exit 1
fi

CSV_FILE="$1"
shift  # Remove first argument, rest are python args
PYTHON_ARGS="$@"

# Check if the CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found"
    exit 1
fi

# Check if sbatch is available
if ! command -v sbatch &> /dev/null; then
    echo "Error: sbatch (Slurm) not found in PATH"
    echo "This script requires Slurm job scheduler"
    exit 1
fi

# Read wandb IDs from CSV file (skip empty lines and comments)
mapfile -t WANDB_IDS < <(grep -v '^#' "$CSV_FILE" | grep -v '^[[:space:]]*$')

if [ ${#WANDB_IDS[@]} -eq 0 ]; then
    echo "No wandb IDs found in '$CSV_FILE'"
    exit 1
fi

echo "Found ${#WANDB_IDS[@]} wandb ID(s) in '$CSV_FILE'"
if [ -n "$PYTHON_ARGS" ]; then
    echo "Additional Python arguments: $PYTHON_ARGS"
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="$SCRIPT_DIR/inference_job_template.sh"

# Check if template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found"
    exit 1
fi

# Create a temporary script for the job array by copying the template
TEMP_SCRIPT=$(mktemp /tmp/testall_XXXXXX.sh)
cp "$TEMPLATE_FILE" "$TEMP_SCRIPT"
CSV_ABS_PATH=$(realpath "$CSV_FILE")

# Make the temporary script executable
chmod +x "$TEMP_SCRIPT"

# Create logs directory if it doesn't exist
mkdir -p /home/jambsdor/projects/DIET_Tuning/logs

# Submit the job array
NUM_JOBS=${#WANDB_IDS[@]}
echo "Submitting job array with $NUM_JOBS tasks..."

if sbatch --array=1-$NUM_JOBS "$TEMP_SCRIPT" "$CSV_ABS_PATH" "$PYTHON_ARGS"; then
    echo "✓ Successfully submitted job array for $NUM_JOBS inference tasks"
    echo "✓ Logs will be saved to /home/jambsdor/projects/DIET_Tuning/logs/inference_<job_id>_<array_id>.{out,err}"
    echo "✓ Monitor jobs with: squeue -u \$USER"
    echo "✓ Cancel all jobs with: scancel <job_id>"
else
    echo "✗ Failed to submit job array"
    rm -f "$TEMP_SCRIPT"
    exit 1
fi

# Clean up temporary script after a delay (gives Slurm time to read it)
(sleep 10 && rm -f "$TEMP_SCRIPT") &

echo ""
echo "Job array submitted successfully!"
echo "CSV file: $CSV_FILE"
echo "Number of tasks: $NUM_JOBS"
echo "Additional args: $PYTHON_ARGS"
