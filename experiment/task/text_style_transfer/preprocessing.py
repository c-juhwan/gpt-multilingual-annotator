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
    data_df, lang_code = load_data(args)

    # Define the tokenizer
    # We're performing text style transfer, so language code is same for both informal and formal text
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=lang_code, tgt_lang=lang_code)

    data_dict = {
        'train': {
            'informal_text': [],
            'formal_text': [],
            'all_references': [],
            'text_number': [],
            'category': [],
            'model_input_ids': [],
            'formal_input_ids': [],
            'tokenizer': tokenizer,
        },
        'valid': {
            'informal_text': [],
            'formal_text': [],
            'all_references': [],
            'text_number': [],
            'category': [],
            'model_input_ids': [],
            'tokenizer': tokenizer,
        },
        'test': {
            'informal_text': [],
            'formal_text': [],
            'all_references': [],
            'text_number': [],
            'category': [],
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
        informal_text = data_df['informal_text'][idx]
        formal_text = data_df['formal_text'][idx]
        text_number = data_df['text_number'][idx]
        category = data_df['category'][idx]
        split = data_df['split'][idx]

        if text_number != 1: # if text_number is not 1, skip the data
            # we only need text_number is 2, 3, 4... for annotated data and test data
            # we will gather the data that has same index later
            continue

        # all_reference: gather all the formal text that has same index
        all_references = data_df[data_df['idx'] == index]['formal_text'].tolist()

        # Tokenize the multi-lingual text
        tokenized = tokenizer(informal_text, text_target=formal_text if split != 'test' else None,
                              padding='max_length', truncation=True,
                              max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to the data_dict
        data_dict[split]['informal_text'].append(informal_text)
        data_dict[split]['formal_text'].append(formal_text)
        data_dict[split]['all_references'].append(all_references)
        data_dict[split]['text_number'].append(text_number)
        data_dict[split]['category'].append(category)
        data_dict[split]['model_input_ids'].append(tokenized['input_ids'].squeeze())

    assert len(data_dict['train']['informal_text']) == len(data_dict['train']['formal_text']) == len(data_dict['train']['all_references']) == len(data_dict['train']['text_number']) == len(data_dict['train']['category']) == len(data_dict['train']['model_input_ids'])

    # Save the data_dict for each split as pickle file
    if args.task_dataset == 'gyafc_en': # Save train only for GYAFC_EN
        assert len(data_dict['train']['informal_text']) == 6000
        with open(os.path.join(preprocessed_path, 'train_ORIGINAL_EN.pkl'), 'wb') as f:
            pickle.dump(data_dict['train'], f)
    elif args.task_dataset == 'xformal_fr':
        assert len(data_dict['train']['informal_text']) == 6000
        assert len(data_dict['valid']['informal_text']) == 100
        assert len(data_dict['test']['informal_text']) == 900
        for split in data_dict.keys():
            with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_FR.pkl'), 'wb') as f:
                pickle.dump(data_dict[split], f)
    elif args.task_dataset == 'xformal_pt':
        assert len(data_dict['train']['informal_text']) == 6000
        assert len(data_dict['valid']['informal_text']) == 100
        assert len(data_dict['test']['informal_text']) == 900
        for split in data_dict.keys():
            with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_PT.pkl'), 'wb') as f:
                pickle.dump(data_dict[split], f)
    elif args.task_dataset == 'xformal_it':
        assert len(data_dict['train']['informal_text']) == 6000
        assert len(data_dict['valid']['informal_text']) == 100
        assert len(data_dict['test']['informal_text']) == 900
        for split in data_dict.keys():
            with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_IT.pkl'), 'wb') as f:
                pickle.dump(data_dict[split], f)

def get_dataset_path(args: argparse.Namespace) -> tuple:
    # Specify the path to the dataset
    # print current working directory

    if args.task_dataset == 'gyafc_en':
        dataset_path = os.path.join(os.getcwd(), 'task/text_style_transfer/XFORMAL/processed', 'gyafc_english.json')
        lang_code = 'en_XX'
    elif args.task_dataset == 'xformal_fr':
        dataset_path = os.path.join(os.getcwd(), 'task/text_style_transfer/XFORMAL/processed', 'xformal_french.json')
        lang_code = 'fr_XX'
    elif args.task_dataset == 'xformal_pt':
        dataset_path = os.path.join(os.getcwd(), 'task/text_style_transfer/XFORMAL/processed', 'xformal_bra_portuguese.json')
        lang_code = 'pt_XX'
    elif args.task_dataset == 'xformal_it':
        dataset_path = os.path.join(os.getcwd(), 'task/text_style_transfer/XFORMAL/processed', 'xformal_italian.json')
        lang_code = 'it_IT'

    return dataset_path, lang_code

def load_data(args: argparse.Namespace) -> pd.DataFrame:
    dataset_path, lang_code = get_dataset_path(args)

    # Load the dataset - read json file into a pandas dataframe
    data_df = pd.read_json(dataset_path, orient='records')

    # Process - Remove '\n' in the text
    data_df['informal_text'] = data_df['informal_text'].apply(lambda x: x.replace('\n', ''))
    data_df['formal_text'] = data_df['formal_text'].apply(lambda x: x.replace('\n', ''))

    return data_df, lang_code
