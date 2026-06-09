#!/bin/bash
#SBATCH --job-name=esm2_gfp_train
#SBATCH --output=../logs/slurm/%x_%j.out
#SBATCH --error=../logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=intbio
#SBATCH --nodelist=node03
#SBATCH --time=24:00:00


conda run -n newNucDPosIT python3 ../scripts/train_model.py