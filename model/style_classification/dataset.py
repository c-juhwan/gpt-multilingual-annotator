# Standard Library Modules
import os
import pickle
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, args: argparse.Namespace, data_path: str, split: str) -> None:
        super(ClassificationDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.args = args
        self.data_list = []
        self.tokenizer = data_['tokenizer']

        for idx in tqdm(range(len(data_['text'])), desc=f'Loading data from {data_path}'):
            # Load data
            text = data_['text'][idx]
            label = data_['label'][idx]
            category = data_['category'][idx]

            self.data_list.append({
                'index': idx,
                'text': text,
                'label': torch.tensor(label, dtype=torch.long), # (1)
                'category': category,
            })

        del data_

    def __getitem__(self, index: int) -> dict:
        return self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    index = [d['index'] for d in data] # list of integers (batch_size)
    text = [d['text'] for d in data] # list of strings (batch_size)
    label = torch.stack([d['label'] for d in data], dim=0) # (batch_size, 1)
    category = [d['category'] for d in data] # list of strings (batch_size)

    datas_dict = {
        'index': index,
        'text': text,
        'label': label,
        'category': category,
    }

    return datas_dict
