# Standard Library Modules
import os
import sys
import time
import pickle
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from easynmt import EasyNMT
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path
from task.captioning.preprocessing import load_caption_data

tmp_lang = ['fr', 'de', 'es', 'ru']

def backtrans_annotating(args: argparse.Namespace) -> None:
    # Define tokenizer - we use bart tokenizer because it has start and end token
    en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    nmt_model = EasyNMT('mbart50_m2m')

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

    # gather only caption_number == 1
    for idx in range(len(loaded_data['caption_numbers'])):
        if loaded_data['caption_numbers'][idx] == 1:
            train_data_dict['image_names'].append(loaded_data['image_names'][idx])
            train_data_dict['caption_numbers'].append(loaded_data['caption_numbers'][idx])
            train_data_dict['captions'].append(loaded_data['captions'][idx])
            train_data_dict['all_captions'].append(loaded_data['all_captions'][idx])
            train_data_dict['input_ids'].append(loaded_data['input_ids'][idx])
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
    for idx in tqdm(range(len(train_data_dict['image_names'])), desc='Annotating with BT...'):
        # Get image_name, caption
        image_name = train_data_dict['image_names'][idx]
        gold_caption = train_data_dict['captions'][idx]

        # Backtranslate
        error_counter = 0
        while True:
            try:
                # Get backtranslated captions
                backtrans_captions = back_translation(nmt_model, gold_caption)

                result_sentences = [gold_caption] + backtrans_captions
            except KeyboardInterrupt as k:
                raise k # if KeyboardInterrupt, raise it to stop the program
            except Exception as e:
                tqdm.write(f"Error {error_counter}: {Exception.__name__} in call_bt: {e}")
                error_counter += 1
                if error_counter > 3:
                    tqdm.write("Error: Too many errors. Skip this image.")
                    break
                else:
                    time.sleep(0.5)
                    continue
            break

        # Tokenize and append to data_dict
        tqdm.write(f"{result_sentences}")
        for i in range(len(result_sentences)):
            # Tokenize
            tokenized = en_tokenizer(result_sentences[i], padding='max_length', truncation=True,
                                     max_length=args.max_seq_len, return_tensors='pt')

            # Append to data_dict
            save_data['image_names'].append(image_name)
            save_data['captions'].append(result_sentences[i])
            save_data['caption_numbers'].append(i+1) # 1 is gold caption
            save_data['input_ids'].append(tokenized['input_ids'].squeeze())

    save_name = 'train_BT_EN.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(save_data, f)
        print(f"Saved {save_name} in {preprocessed_path}")
        print(len(save_data['image_names']))

def back_translation(model, src: str, num: int=4) -> list: # list of str (backtranslated captions)
    mid_result = [model.translate(src, target_lang=each_tmp_lang) for each_tmp_lang in tmp_lang]
    result = [model.translate(each_mid_result, target_lang='en') for each_mid_result in mid_result]

    return result
