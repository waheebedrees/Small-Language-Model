import os
import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

class DataLoader:
    def __init__(self, batch_size, block_size, device='cpu'):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.enc = tiktoken.get_encoding("gpt2")
        
    def prepare_data(self, dataset_name="roneneldan/TinyStories", force_reload=False):
        """
        Prepare and tokenize the dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            force_reload: Whether to force reload even if files exist
        """
        if not os.path.exists("train.bin") or force_reload:
            print("Loading dataset...")
            ds = load_dataset(dataset_name)
            
            def process(example):
                ids = self.enc.encode_ordinary(example['text'])
                return {'ids': ids, 'len': len(ids)}
            
            print("Tokenizing dataset...")
            tokenized = ds.map(
                process,
                remove_columns=['text'],
                desc="tokenizing the splits",
                num_proc=8,
            )
            
            # Save tokenized data to binary files
            for split, dset in tokenized.items():
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                filename = f'{split}.bin'
                dtype = np.uint16
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
                total_batches = 1024

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()
                
            print("Data preparation complete!")
        else:
            print("Using existing tokenized data files.")
    
    def get_batch(self, split='train'):
        """
        Get a batch of data for training or validation.
        
        Args:
            split: 'train' or 'validation'
            
        Returns:
            x, y: Input and target tensors
        """
        if split == 'train':
            data = np.memmap('train.bin', dtype=np.uint16, mode='r')
        else:
            data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
            
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if 'cuda' in self.device:
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y
    
    def encode(self, text):
        """Encode text to tokens."""
        return self.enc.encode_ordinary(text)
    
    def decode(self, tokens):
        """Decode tokens to text."""
        return self.enc.decode(tokens)

