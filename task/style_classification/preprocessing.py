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
from PIL import Image
from pycocotools.coco import COCO
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def preprocessing(args: argparse.Namespace) -> None:
    # Load the dataset
    data_df, lang_code, out_lang_code = load_data(args)

    # Define the tokenizer
    # We're performing text style transfer, so language code is same for both informal and formal text
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=lang_code, tgt_lang=lang_code)

    data_dict = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
            'tokenizer': tokenizer,
        },
        'valid': {
            'text': [],
            'label': [],
            'category': [],
            'tokenizer': tokenizer,
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
            'tokenizer': tokenizer,
        },
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)

    for idx in tqdm(range(len(data_df)), desc='Preprocessing data...'):
        text = data_df['text'][idx]
        label = data_df['label'][idx]
        category = data_df['category'][idx]
        split = data_df['split'][idx]

        # Append the data to the data_dict
        data_dict[split]['text'].append(text)
        data_dict[split]['label'].append(label)
        data_dict[split]['category'].append(category)

    # Save the data_dict for each split as pickle file
    for split in data_dict.keys():
        with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_{out_lang_code}.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

def get_dataset_path(args: argparse.Namespace) -> tuple:

    if args.task_dataset == 'xformal_fr':
        dataset_path = os.path.join(os.getcwd(), 'task/style_classification/XFORMAL/processed', 'xformal_french.json')
        lang_code = 'fr_XX'
        out_lang_code = 'FR'
    elif args.task_dataset == 'xformal_pt':
        dataset_path = os.path.join(os.getcwd(), 'task/style_classification/XFORMAL/processed', 'xformal_bra_portuguese.json')
        lang_code = 'pt_XX'
        out_lang_code = 'PT'
    elif args.task_dataset == 'xformal_it':
        dataset_path = os.path.join(os.getcwd(), 'task/style_classification/XFORMAL/processed', 'xformal_italian.json')
        lang_code = 'it_IT'
        out_lang_code = 'IT'

    return dataset_path, lang_code, out_lang_code

def load_data(args: argparse.Namespace) -> pd.DataFrame:
    dataset_path, lang_code, out_lang_code = get_dataset_path(args)

    # Load the dataset
    data_df = pd.read_json(dataset_path, orient='records')

    # Process - Remove '\n' in the text
    data_df['text'] = data_df['text'].apply(lambda x: x.replace('\n', ''))

    return data_df, lang_code, out_lang_code
