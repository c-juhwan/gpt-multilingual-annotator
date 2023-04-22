# Standard Library Modules
import os
import sys
import time
import pickle
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
import re
import random
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path
from task.captioning.preprocessing import load_caption_data

def onlyone_annotating(args: argparse.Namespace) -> None:
    # Load caption data
    caption_df = load_caption_data(args)

    # Define tokenizer - we use bart tokenizer because it has start and end token
    en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    # Define data_dict
    data_dict_en = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
    check_path(preprocessed_path)

    # for split == 0, only remain caption_number == 1
    train_df = caption_df[caption_df['split'] == 0]
    train_df.reset_index(drop=True, inplace=True)

    # Remain only 1 caption per each image_name
    train_df = train_df.groupby('image_name').first().reset_index()
    print(train_df)

    for idx in tqdm(range(len(train_df)), desc='Annotating with only one sample...'):
        # Get image_name, caption
        image_name = caption_df['image_name'][idx]
        gold_caption = caption_df['caption_text'][idx]

        # Tokenize
        tokenized = en_tokenizer(gold_caption, padding='max_length', truncation=True,
                                 max_length=args.max_seq_len, return_tensors='pt')

        # Append to data_dict
        data_dict_en['image_names'].append(image_name)
        data_dict_en['captions'].append(gold_caption)
        data_dict_en['caption_numbers'].append(1)
        data_dict_en['input_ids'].append(tokenized['input_ids'].squeeze())

    save_name = 'train_ONE_EN.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(data_dict_en, f)
        print(f'Saved {save_name} at {preprocessed_path}')