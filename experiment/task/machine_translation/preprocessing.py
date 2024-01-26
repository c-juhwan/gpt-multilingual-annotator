# Standard Library Modules
import os
import sys
import json
import pickle
import random
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def preprocessing(args: argparse.Namespace) -> None:
    # Load the dataset
    data_df = load_data(args)

    # Define the tokenizer

    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='de_DE')

    data_dict = {
        'train': {
            'source_text': [],
            'target_text': [],
            'text_number': [],
            'model_input_ids': [],
            'tokenizer': tokenizer,
        },
        'valid': {
            'source_text': [],
            'target_text': [],
            'text_number': [],
            'model_input_ids': [],
            'tokenizer': tokenizer,
        },
        'test': {
            'source_text': [],
            'target_text': [],
            'text_number': [],
            'model_input_ids': [],
            'tokenizer': tokenizer,
        },
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)

    for idx in tqdm(range(len(data_df)), desc='Preprocessing data...'):
        # Get the data from the dataframe
        index = data_df['idx'][idx]
        source_text = data_df['source_text'][idx]
        target_text = data_df['target_text'][idx]
        text_number = data_df['text_number'][idx]
        split = data_df['split'][idx]

        # Tokenize the multi-lingual text
        tokenized = tokenizer(source_text, text_target=target_text if split != 'test' else None,
                              padding='max_length', truncation=True,
                              max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to the data_dict
        data_dict[split]['source_text'].append(source_text)
        data_dict[split]['target_text'].append(target_text)
        data_dict[split]['text_number'].append(text_number)
        data_dict[split]['model_input_ids'].append(tokenized['input_ids'].squeeze())

    # Save the data_dict for each split as pickle file
    for split in data_dict.keys():
        with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_DE.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

def load_data(args: argparse.Namespace) -> pd.DataFrame:
    dataset = load_dataset('bentrevett/multi30k')

    data_df = pd.DataFrame(columns=['idx', 'source_text', 'target_text', 'text_number', 'split'])

    idx = 0
    for split in ['train', 'validation', 'test']:
        for idx in tqdm(range(len(dataset[split])), desc=f'Loading {split} data...'):
            # Get the data from the dataset
            index = idx
            source_text = dataset[split]['en'][idx]
            target_text = dataset[split]['de'][idx]
            text_number = 1

            # Append the data to the dataframe
            data_df = data_df.append({'idx': index, 'source_text': source_text, 'target_text': target_text, 'text_number': text_number, 'split': split}, ignore_index=True)

    # Process - Remove '\n' in the text
    data_df['source_text'] = data_df['source_text'].apply(lambda x: x.replace('\n', ''))
    data_df['target_text'] = data_df['target_text'].apply(lambda x: x.replace('\n', ''))

    # 'Validation' -> 'valid'
    data_df['split'] = data_df['split'].apply(lambda x: 'valid' if x == 'validation' else x)

    return data_df
