from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable, Iterable, List, Optional, Sequence, Union


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


"""
Dataset / DataLoader for ProteinMPNN outputs.
  - Lazy I/O: build a byte-offset index in parallel, read one sequence per __getitem__
  - Optional preload: load everything into RAM in parallel chunks (fast inference, high RAM)
  - DataLoader factory: num_workers prefetch for GPU-bound scoring pipelines
"""


FastaRecord = namedtuple("FastaRecord", ("header", "sequence", "filepath", "index"))


def _resolve_fasta_files(fasta_paths: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(fasta_paths, str):
        if os.path.isdir(fasta_paths):
            root = fasta_paths
            return sorted(
                os.path.join(root, name)
                for name in os.listdir(root)
                if name.endswith((".fa", ".fasta"))
            )
        return [fasta_paths]

    return [str(path) for path in fasta_paths]


def _index_fasta_file(filepath: str) -> List[dict]:
    """Return per-sequence byte offsets for lazy reads."""
    entries: List[dict] = []
    header: Optional[str] = None
    seq_start: Optional[int] = None

    with open(filepath, "rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break

            if line.startswith(b">"):
                if header is not None and seq_start is not None:
                    entries.append(
                        {
                            "filepath": filepath,
                            "header": header,
                            "offset": seq_start,
                            "end": line_start,
                        }
                    )
                header = line.decode("utf-8", errors="replace").strip()[1:]
                seq_start = handle.tell()
            elif header is None:
                continue

        if header is not None and seq_start is not None:
            entries.append(
                {
                    "filepath": filepath,
                    "header": header,
                    "offset": seq_start,
                    "end": handle.tell(),
                }
            )

    return entries


def _read_fasta_entry(filepath: str, offset: int, end: int) -> str:
    with open(filepath, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(end - offset)
    return raw.replace(b"\n", b"").decode("utf-8", errors="replace")


def _default_index_workers() -> int:
    cpu = os.cpu_count() or 1
    return min(8, cpu)


class FastaDataset(Dataset):
    """
    FASTA dataset with parallel index construction and lazy or preloaded sequence access.

    Parameters
    ----------
    fasta_paths :
        Directory of *.fa/*.fasta files, a single file, or an explicit list of paths.
    preload :
        If True, load all sequences into RAM after indexing (best throughput when memory allows).
    index_workers :
        Processes used while building the file index. Defaults to min(8, cpu_count()).
    preload_workers :
        Threads used while preloading sequences. Defaults to min(16, cpu_count() * 2).
    preload_chunk_size :
        Number of sequences loaded per preload wave to bound peak memory during preload.
    transform :
        Optional callable applied to each sequence string before returning.
    return_header :
        If True, __getitem__ returns FastaRecord; otherwise returns the sequence string only.
    """

    def __init__(
        self,
        fasta_paths: Union[str, Sequence[str]],
        *,
        preload: bool = False,
        index_workers: Optional[int] = None,
        preload_workers: Optional[int] = None,
        preload_chunk_size: int = 10_000,
        transform: Optional[Callable[[str], object]] = None,
        return_header: bool = True,
    ):
        self.transform = transform
        self.return_header = return_header
        self.preload = preload

        fasta_files = _resolve_fasta_files(fasta_paths)
        if not fasta_files:
            raise ValueError(f"No FASTA files found for: {fasta_paths!r}")

        index_workers = index_workers or _default_index_workers()
        self._entries = self._build_index_parallel(fasta_files, index_workers)

        if not self._entries:
            raise ValueError(f"No sequences found in FASTA inputs: {fasta_paths!r}")

        for global_idx, entry in enumerate(self._entries):
            entry["index"] = global_idx

        self._preloaded: Optional[List[str]] = None
        if preload:
            preload_workers = preload_workers or min(16, (os.cpu_count() or 1) * 2)
            self._preloaded = self._preload_sequences_parallel(
                preload_workers, preload_chunk_size
            )

    @staticmethod
    def _build_index_parallel(fasta_files: Sequence[str], index_workers: int) -> List[dict]:
        if len(fasta_files) == 1 or index_workers <= 1:
            entries: List[dict] = []
            for filepath in fasta_files:
                entries.extend(_index_fasta_file(filepath))
            return entries

        entries = []
        with ProcessPoolExecutor(max_workers=min(index_workers, len(fasta_files))) as pool:
            for file_entries in pool.map(_index_fasta_file, fasta_files, chunksize=1):
                entries.extend(file_entries)
        return entries

    def _preload_sequences_parallel(self, preload_workers: int, chunk_size: int) -> List[str]:
        sequences = [""] * len(self._entries)
        load_one = partial(self._load_entry_sequence)

        for chunk_start in range(0, len(self._entries), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(self._entries))
            with ThreadPoolExecutor(max_workers=preload_workers) as pool:
                futures = {
                    pool.submit(load_one, idx): idx
                    for idx in range(chunk_start, chunk_end)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    sequences[idx] = future.result()
        return sequences

    def _load_entry_sequence(self, idx: int) -> str:
        entry = self._entries[idx]
        return _read_fasta_entry(entry["filepath"], entry["offset"], entry["end"])

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int):
        if not 0 <= idx < len(self._entries):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        entry = self._entries[idx]
        sequence = self._preloaded[idx] if self._preloaded is not None else self._load_entry_sequence(idx)

        if self.transform is not None:
            sequence = self.transform(sequence)

        if not self.return_header:
            return sequence

        if isinstance(sequence, str):
            return FastaRecord(
                header=entry["header"],
                sequence=sequence,
                filepath=entry["filepath"],
                index=entry["index"],
            )

        return FastaRecord(
            header=entry["header"],
            sequence=sequence,
            filepath=entry["filepath"],
            index=entry["index"],
        )


def fasta_collate(records: Sequence[FastaRecord]) -> dict:
    """Default batch collate: stack metadata, keep sequences as a list for variable-length encoding."""
    return {
        "headers": [record.header for record in records],
        "sequences": [record.sequence for record in records],
        "filepaths": [record.filepath for record in records],
        "indices": torch.tensor([record.index for record in records], dtype=torch.long),
    }


def create_fasta_dataloader(
    fasta_paths: Union[str, Sequence[str]],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = fasta_collate,
    dataset_kwargs: Optional[dict] = None,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Build a DataLoader tuned for large FASTA corpora.

    Lazy mode (preload=False):
      - Low memory at startup; parallel disk reads via DataLoader workers.
    Preload mode (dataset_kwargs={'preload': True}):
      - Higher RAM, minimal I/O during inference; use num_workers mainly for batch assembly.
    """
    dataset_kwargs = dataset_kwargs or {}
    dataset = FastaDataset(fasta_paths, **dataset_kwargs)

    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    loader_kwargs.update(dataloader_kwargs)
    return DataLoader(dataset, **loader_kwargs)

        
