import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os


class FastaDataset(Dataset):
    def __init__(self, fasta_paths):
        """
        fasta_paths: путь к папке с *.fasta, либо один файл, либо список файлов
        """
        if isinstance(fasta_paths, str):
            if os.path.isdir(fasta_paths):
                fasta_files = [
                    os.path.join(fasta_paths, f)
                    for f in os.listdir(fasta_paths)
                    if f.endswith(('.fa', '.fasta'))
                ]
            else:
                fasta_files = [fasta_paths]
        else:
            fasta_files = list(fasta_paths)

        self.sequences = []
        for fpath in fasta_files:
            self.sequences.extend(self._read_fasta(fpath))

    def _read_fasta(self, filepath):
        seqs = []
        current_seq = None
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq is not None:
                        seqs.append(current_seq)
                    current_seq = ''
                else:
                    current_seq += line
            if current_seq is not None:
                seqs.append(current_seq)
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]