import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class SimpleDataset(Dataset):    
    def __init__(self, items, targets):
        self.seqs = items
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


def collate_fn(batch, tokenizer, max_length=1024):
    sequences = [item[0] for item in batch] 
    targets = torch.tensor([item[1] for item in batch])
    
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    return inputs, targets 

        
