#!/bin/bash

#SBATCH --job-name=enrich_data
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_prod_night
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load Python module or activate your Python environment
# For example:
# module load python/3.8
# or
# source mypythonenv/bin/activate

# Activate the venv : 
source /usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_24/project/deep-sdm-venv/bin/activate

# Run your Python script
python enrich_data.py
