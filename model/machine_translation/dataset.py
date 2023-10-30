# Standard Library Modules
import os
import pickle
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset

class MTDataset(Dataset):
    def __init__(self, args: argparse.Namespace, data_path: str, split: str) -> None:
        super(MTDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.tokenizer = data_['tokenizer']

        for idx in tqdm(range(len(data_['source_text'])), desc=f'Loading data from {data_path}'):
            # Load data
            source_text = data_['source_text'][idx]
            target_text = data_['target_text'][idx]
            text_number = data_['text_number'][idx]
            model_input_ids = data_['model_input_ids'][idx]

            self.data_list.append({
                'index': idx,
                'source_text': source_text,
                'target_text': target_text,
                'text_number': text_number,
                'model_input_ids': model_input_ids,
            })

        del data_

    def __getitem__(self, index: int) -> dict:
        return self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    index = [d['index'] for d in data] # list of integers (batch_size)
    source_text = [d['source_text'] for d in data] # list of strings (batch_size)
    target_text = [d['target_text'] for d in data] # list of strings (batch_size)
    text_number = [d['text_number'] for d in data] # list of integers (batch_size)
    model_input_ids = torch.stack([d['model_input_ids'] for d in data], dim=0) # (batch_size, max_seq_len)

    datas_dict = {
        'index': index,
        'source_text': source_text,
        'target_text': target_text,
        'text_number': text_number,
        'model_input_ids': model_input_ids,
    }

    return datas_dict
