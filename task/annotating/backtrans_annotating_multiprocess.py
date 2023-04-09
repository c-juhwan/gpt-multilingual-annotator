# Standard Library Modules
import os
import sys
import time
import pickle
import random
import logging
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
from multiprocessing import Pool
# 3rd-party Modules
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from BackTranslation import BackTranslation
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path
from task.captioning.preprocessing import load_caption_data

en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
NUM_PROCESS = 4
tqdm_bar = tqdm(total=100, desc='Progress', position=0)
tmp_lang = ['fr', 'de', 'es', 'ja']

def backtrans_annotating_multiprocess(args: argparse.Namespace) -> None:
    # Load caption data
    caption_df = load_caption_data(args)

    # Define BackTranslators
    random_port = [
        (random.randint(1000, 9999), random.randint(1000, 9999)) for _ in range(NUM_PROCESS)
    ]
    back_translators = [
        # Has to be different port for each process because of multiprocessing
        BackTranslation(url=['translate.google.com', 'translate.google.co.kr'],
                        proxies={'http': f'127.0.0.1:{port[0]}', 'http://host.name': f'127.0.0.1:{port[1]}'}) for port in random_port
    ]

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
    preprocessed_path = os.path.join(args.preprocess_path, args.task_dataset)
    check_path(preprocessed_path)

    # for split == 0, only remain caption_number == 1
    train_df = caption_df[caption_df['split'] == 0]
    # Remain only 1 caption per each image_name
    train_df = train_df.groupby('image_name').first().reset_index()
    train_df.reset_index(drop=True, inplace=True)

    # Call multiprocessing using starmap
    # Divide train_df into NUM_PROCESS parts
    train_df_subset = np.array_split(train_df, NUM_PROCESS)
    tqdm_bar.total = len(train_df) // NUM_PROCESS

    # Reset index of train_df_subset
    for i in range(NUM_PROCESS):
        train_df_subset[i].reset_index(drop=True, inplace=True)

    # Call multiprocessing
    starmap_items = [
        (args, train_df_subset[i]) for i in range(NUM_PROCESS)
    ]

    print(f"Start multiprocessing with {NUM_PROCESS} processes")

    with Pool(NUM_PROCESS) as p:
       results = p.starmap(try_call_bt, starmap_items)


    print("Done with multiprocessing")

    for result in results:
        data_dict_en['image_names'] += result[0]['image_names'] # result[0] is data_dict_en
        data_dict_en['captions'] += result[0]['captions']
        data_dict_en['all_captions'] += result[0]['all_captions']
        data_dict_en['caption_numbers'] += result[0]['caption_numbers']
        data_dict_en['input_ids'] += result[0]['input_ids']

    save_name = 'train_BT_EN.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(data_dict_en, f)
        print(f"Saved {save_name} in {preprocessed_path}")

    tqdm_bar.close()

def try_call_bt(args: argparse.Namespace, train_df_subset: pd.DataFrame, BackTranslator) -> dict:
    try:
        return call_bt(args, train_df_subset, BackTranslator)
    except KeyboardInterrupt as k:
        raise k
    except Exception as e:
        logging.exception(f"Error in try_call_bt: {e}")

def call_bt(args: argparse.Namespace, train_df_subset: pd.DataFrame, BackTranslator) -> dict:
    subset_dict_en = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }

    for idx in tqdm(range(len(train_df_subset)), desc='Annotating with BackTranslation...'):
        # Get image_name, caption
        image_name = train_df_subset['image_name'][idx]
        gold_caption = train_df_subset['caption_text'][idx]

        # Backtranslate
        error_counter = 0
        while True:
            try:
                # Get backtranslated captions
                backtrans_captions = [
                    BackTranslator.translate(gold_caption, src='en', tmp=lang).result_text for lang in tmp_lang
                ] # Save 4 backtranslated captions for each gold caption

                result_sentences = [gold_caption] + backtrans_captions
            except KeyboardInterrupt as k:
                raise k # if KeyboardInterrupt, raise it to stop the program
            except Exception as e:
                #print(f"Error {error_counter}: {Exception}")
                error_counter += 1
                if error_counter > 3:
                    #print("Error: Too many errors. Skip this image.")
                    break
                else:
                    time.sleep(0.5)
                    continue
            break

        # Tokenize and append to data_dict
        for i in range(len(result_sentences)):
            # Tokenize
            tokenized = en_tokenizer(result_sentences[i], padding='max_length', truncation=True,
                                     max_length=args.max_seq_len, return_tensors='pt')

            # Append to data_dict
            subset_dict_en['image_names'].append(image_name)
            subset_dict_en['captions'].append(result_sentences[i])
            subset_dict_en['caption_numbers'].append(i+1)
            subset_dict_en['input_ids'].append(tokenized['input_ids'].squeeze())
            subset_dict_en['all_captions'].append(result_sentences)

        tqdm_bar.update(1)

    return subset_dict_en
