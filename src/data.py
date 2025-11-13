# src/data.py

import os
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CharTokenizer:
    """
    Simple character-level tokenizer:
    - builds vocab from text
    - encodes text -> list[int]
    - decodes list[int] -> text
    """
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str):
        """String -> list of ids"""
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """List of ids -> string"""
        return ''.join([self.itos[i] for i in ids])


class CharDataset(Dataset):
    """
    Dataset of overlapping sequences for next-token prediction.
    Given a long sequence of token ids, we create pairs:
        x = ids[i : i+block_size]
        y = ids[i+1 : i+1+block_size]
    """
    def __init__(self, data: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # len(data) - block_size হতে যদি negative আসে,
        # সেটাকে 0 করে দিচ্ছি যাতে DataLoader error না দেয়।
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def load_data(
    corpus_path: str,
    block_size: int = 64,
    batch_size: int = 32,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    """
    Reads corpus.txt, builds tokenizer, creates train & val DataLoaders.

    Returns:
        train_loader, val_loader, tokenizer
    """
    assert os.path.isfile(corpus_path), f"Corpus file not found: {corpus_path}"

    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # build tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Loaded corpus with {len(text)} characters, vocab size = {tokenizer.vocab_size}")

    # encode entire text
    data_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # train/val split
    n = int(len(data_ids) * train_split)
    train_data = data_ids[:n]
    val_data = data_ids[n:]

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,    # train এ small last batch ফেলে দিচ্ছি
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,   # validation এ small batch retain করছি
    )

    return train_loader, val_loader, tokenizer
