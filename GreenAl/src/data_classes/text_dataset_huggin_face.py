from datasets import load_from_disk

import argparse
import os

from datasets import load_dataset
from transformers import TextDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.multiprocessing as mp

from torch.utils.data import Dataset
import sys


def flatten(t):
    return [item for sublist in t for item in sublist]

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

class TextDatasetHugginFace(Dataset):
    """
    A map-style dataset that can be used for big data (~millions of objects). 
    Building objects is similar to transformers.TextDataset
    Attributes:
    ----------
    tokenizer: PreTrainedTokenizer
        Tokenizer for processing text
    block_size: int
        The length of the tokenized text
    dataset: datasets.Dataset
        Datase for tokenization, obtained by datasets.load_dataset() (if has_tokens = False)
    has_tokens: bool
        Are texts already tokenized
    k: int
        A number of text objects loading and processing together.
    tokens_path: str
        Path to preprocessed tokens (if has_tokens = True)
    """
    
    
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        block_size: int,
        has_tokens = True,
        k = 100,
        tokens_path ='/home/jovyan/greenAI_gpt/openwebtext2/train.tokens'
    ):
    """
    Build the Dataset object with a list of tokenized texts of size=block_size, by loading and processing k texts simultaneously.
    """
        self.examples = []
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        if (not has_tokens):
            print ("tokenization start")
            dataset_train = dataset.map(tokenize_function, batched=True)
            tokenized_text = dataset_train
            tokenized_text.save_to_disk(tokens_path)
            print ("tokenization end")
        tokenized_text = load_from_disk(tokens_path)
        dataset_block_size = block_size*k
        for j in range(0, len(tokenized_text) - dataset_block_size + 1, dataset_block_size):
            block_dataset = flatten(tokenized_text[j : j + dataset_block_size]['input_ids'])
            trash = 500000
            if (j > trash):
                print (j, flush = True)
                trash +=trash
            for i in range(0, len(block_dataset) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(block_dataset[i : i + block_size]))
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
