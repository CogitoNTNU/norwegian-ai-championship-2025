#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=tumor_segmentation
#SBATCH --time=5:00:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a CPU
#SBATCH --mem=32G                 # Asking for xGB RAM
#SBATCH --constraint="(v100|a100|h100)&(gpu32g|gpu40g|gpu80g)"
#SBATCH --nodes=1
#SBATCH --output=idun-log.txt      # Specifying ‘stdout’
#SBATCH --error=idun-log.err        # Specifying ‘stderr’
#SBATCH --mail-user=minaoa@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/work/sverrnys/NM-ai/norwegian-ai-championship-2025/cv-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.11

# Activate the environment
conda activate ${ENV_PATH}

# UV
pip install uv
uv sync

uv run python train_unet_plus_plus_kfold_backbone.py