#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=RL-carRacing
#SBATCH --time=0-03:00:00         # format: D-HH:MM:SS
#SBATCH --partition=CPUQ          # Asking for a GPU
#SBATCH --mem=8G                 # Asking for xGB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 # x CPU cores
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

ENV_PATH="/cluster/work/sverrnys/hpc-slurm-course/cv-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.11

# Activate the environment
conda activate ${ENV_PATH}

# UV
pip install uv

python main.py