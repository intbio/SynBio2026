#!/bin/sh
#SBATCH --job-name=Synbio2026_score_seqsx2GPU
#SBATCH --output=../logs/slurm/%x_%j.out
#SBATCH --error=../logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_rtx_a5000:2
#SBATCH --partition=intbio
#SBATCH --nodelist=node14
#SBATCH --time=24:00:00

# Запускаем первую задачу в фоне с GPU 0
conda run -n newNucDPosIT python3 ../scripts/score_sequences.py \
    --input_dir ../data/mpnn_sequences/HyperMPNN/batch1 \
    --output ../data/scored_sequences/scores_batch1.csv \
    --batch_size 1000 \
    --num_workers 4 \
    --device cuda:0 &

# Запускаем вторую задачу в фоне с GPU 1
conda run -n newNucDPosIT python3 ../scripts/score_sequences.py \
    --input_dir ../data/mpnn_sequences/HyperMPNN/batch2 \
    --output ../data/scored_sequences/scores_batch2.csv \
    --batch_size 1000 \
    --num_workers 4 \
    --device cuda:1 &

# Ждём завершения ОБОИХ процессов
wait

echo "Обе задачи завершены!"