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

def budget_annotating(args: argparse.Namespace) -> None:
    # Define tokenizer - we use bart tokenizer because it has start and end token
    en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    # Define data_dict
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_ORIGINAL_EN.pkl'), 'rb') as f:
        loaded_data = pickle.load(f)

    train_data_dict = {
        'image_names': [],
        'caption_numbers': [],
        'captions': [],
        'all_captions': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }
    if args.task_dataset == 'flickr8k':
        budget_limit = 1500 * 5 # 1500 images * 5 captions, refer to the paper
    elif args.task_dataset == 'flickr30k':
        budget_limit = 5800 * 5 # 5800 images * 5 captions, refer to the paper
    elif args.task_dataset == 'coco2014':
        budget_limit = 19680 * 5 # 19680 images * 5 captions, refer to the paper
    else:
        raise NotImplementedError

    # gather within budget limit
    counter = 0
    for idx in range(budget_limit):
        train_data_dict['image_names'].append(loaded_data['image_names'][idx])
        train_data_dict['caption_numbers'].append(loaded_data['caption_numbers'][idx])
        train_data_dict['captions'].append(loaded_data['captions'][idx])
        train_data_dict['all_captions'].append(loaded_data['all_captions'][idx])
        train_data_dict['input_ids'].append(loaded_data['input_ids'][idx])
        if loaded_data['caption_numbers'][idx] == 1:
            counter += 1
        else:
            continue

    save_data = {
        'image_names': [],
        'caption_numbers': [],
        'captions': [],
        'all_captions': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
    check_path(preprocessed_path)
    for idx in tqdm(range(len(train_data_dict['image_names'])), desc='Annotating with limited budget...'):
        # Get image_name, caption
        image_name = train_data_dict['image_names'][idx]
        gold_caption = train_data_dict['captions'][idx]
        caption_number = train_data_dict['caption_numbers'][idx]

        # Tokenize
        tokenized = en_tokenizer(gold_caption, padding='max_length', truncation=True,
                                 max_length=args.max_seq_len, return_tensors='pt')

        # Append to data_dict
        save_data['image_names'].append(image_name)
        save_data['captions'].append(gold_caption)
        save_data['caption_numbers'].append(caption_number)
        save_data['input_ids'].append(tokenized['input_ids'].squeeze())

    save_name = 'train_BUD_EN.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(save_data, f)
        print(f'Saved {save_name} at {preprocessed_path}')
