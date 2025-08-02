#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=RL-carRacing
#SBATCH --time=0-03:00:00         # format: D-HH:MM:SS
#SBATCH --partition=CPUQ          # Asking for a GPU
#SBATCH --mem=8G                 # Asking for xGB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30 # x CPU cores
#SBATCH --output=idun-log.txt      # Specifying ‘stdout’
#SBATCH --error=idun-log.err        # Specifying ‘stderr’
#SBATCH --mail-user=chrisfjo@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/home/chrisfjo/NM-ai/norwegian-ai-championship-2025/cv-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.11

# Activate the environment
conda activate ${ENV_PATH}

# UV
pip install uv
uv sync

export WANDB_API_KEY=322fddbf7912adefb6bc85d1bfd7330515e2c042
python -c "import wandb; wandb.login(); print(wandb.api.api_key)"
uv run python DM-i-AI-2025/race-car/train_ppo_real.py --rounds 10000