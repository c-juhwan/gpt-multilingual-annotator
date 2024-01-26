# Standard Library Modules
import os
import pickle
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset

class TSTDataset(Dataset):
    def __init__(self, args: argparse.Namespace, data_path: str, split: str) -> None:
        super(TSTDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.tokenizer = data_['tokenizer']

        for idx in tqdm(range(len(data_['informal_text'])), desc=f'Loading data from {data_path}'):
            # Load data
            informal_text = data_['informal_text'][idx]
            formal_text = data_['formal_text'][idx]
            all_references = data_['all_references'][idx]
            text_number = data_['text_number'][idx]
            category = data_['category'][idx]
            model_input_ids = data_['model_input_ids'][idx]

            self.data_list.append({
                'index': idx,
                'informal_text': informal_text,
                'formal_text': formal_text,
                'all_references': all_references,
                'text_number': text_number,
                'category': category,
                'model_input_ids': model_input_ids,
            })

        del data_

    def __getitem__(self, index: int) -> dict:
        return self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    index = [d['index'] for d in data] # list of integers (batch_size)
    informal_text = [d['informal_text'] for d in data] # list of strings (batch_size)
    formal_text = [d['formal_text'] for d in data] # list of strings (batch_size)
    all_references = [d['all_references'] for d in data] # list of list of strings (batch_size, num_references)
    text_number = [d['text_number'] for d in data] # list of integers (batch_size)
    category = [d['category'] for d in data] # list of strings (batch_size)
    model_input_ids = torch.stack([d['model_input_ids'] for d in data], dim=0) # (batch_size, max_seq_len)

    datas_dict = {
        'index': index,
        'informal_text': informal_text,
        'formal_text': formal_text,
        'all_references': all_references,
        'text_number': text_number,
        'category': category,
        'model_input_ids': model_input_ids,
    }

    return datas_dict
