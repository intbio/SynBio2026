#!/bin/bash
#SBATCH --job-name=Synbio2026_score_seqs
#SBATCH --output=../logs/slurm/%x_%j.out
#SBATCH --error=../logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --partition=intbio
#SBATCH --nodelist=node14
#SBATCH --time=24:00:00


conda run -n newNucDPosIT python3 ../scripts/score_sequences.py