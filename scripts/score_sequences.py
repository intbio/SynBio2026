import sys, os
sys.path.append('../src/FluoreModel/')

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import glob
import csv
import gc
from torch.nn.parallel import DistributedDataParallel as DDP

from reslinear_model import ResLinear
from model import GFPRegressionModel
from gfp_datasets import create_fasta_dataloader
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='GFP brightness prediction from FASTA files')
    
    # Пути
    parser.add_argument('--model_weights', type=str, 
                        default='../data/model_weights/regression_model.weights',
                        help='Path to model weights file')
    parser.add_argument('--input_dir', type=str,
                        default='../data/mpnn_sequences/MPNN_seqs/',
                        help='Directory containing .fa files')
    parser.add_argument('--output', type=str,
                        default='../data/scored_sequences/scores.csv',
                        help='Output CSV file path')
    
    parser.add_argument('--batch_size', type=int, default=1500,
                        help='Batch size for inference')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='Buffer size for batch writing')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers')
    
    # Параметры устройства
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


args = parse_args()

# Параметры
MODEL_WEIGHTS_PATH = args.model_weights
ESM_TYPE = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = args.batch_size
OUTPATH = args.output
BUFFER_SIZE = args.buffer_size 
DEVICE = args.device

# Очистка кэша перед загрузкой
torch.cuda.empty_cache()
gc.collect()

# Загрузка модели
brightness_model = ResLinear(1280, 10)
brightness_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
brightness_model = brightness_model.eval()

# Перемещение модели в GPU
brightness_model = brightness_model.to(DEVICE)

GFPModel = GFPRegressionModel(None, brightness_model, device=DEVICE)
GFPModel = GFPModel.eval()

# Оптимизация токенизатора
tokenizer = AutoTokenizer.from_pretrained(
    ESM_TYPE, 
    model_max_length=1024  # ограничение длины
)

fasta_files = list(Path(args.input_dir).glob('*.fa'))

loader = create_fasta_dataloader(
    fasta_files,      
    batch_size=BATCH_SIZE,
    num_workers=args.num_workers,  # уменьшено
    shuffle=False,
    dataset_kwargs={"preload": False},
)

file_exists = os.path.isfile(OUTPATH)
with open(OUTPATH, 'a' if file_exists else 'w', newline='') as f:
    writer = csv.writer(f)
    
    if not file_exists:
        writer.writerow(['id', 'struct', 'brightness', 'sequence'])
    
    buffer = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            sequences = batch['sequences']
            
            # Токенизация с пакетной обработкой
            tokens = tokenizer(
                sequences, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=1024
            )
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            
            # Использование смешанной точности
            with torch.cuda.amp.autocast():
                predicted_brightness = GFPModel(tokens)
            
            predicted_brightness = predicted_brightness.cpu().numpy().flatten()
            structs = [record.split(' ')[0] for record in batch['headers']]
            ids = batch['indices'].numpy()
            
            for i in range(len(predicted_brightness)):
                buffer.append([ids[i], structs[i], predicted_brightness[i], sequences[i]])
            
            # Очистка GPU памяти
            del tokens
            del predicted_brightness
            torch.cuda.empty_cache()
            
            # Запись буфера
            if len(buffer) >= BUFFER_SIZE:
                writer.writerows(buffer)
                f.flush()
                buffer.clear()
                gc.collect()
            
            # Периодическая очистка каждые 10 батчей
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            print(f"Processed batch {batch_idx+1}, memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    if buffer:
        writer.writerows(buffer)
        f.flush()

print("Done!")