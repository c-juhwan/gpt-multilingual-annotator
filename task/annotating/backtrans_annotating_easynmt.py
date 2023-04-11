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

#tmp_lang = ['fr', 'de', 'es', 'it', 'nl', 'ru', 'zh', 'jap']
tmp_lang = ['fr', 'de', 'es', 'ru']

def backtrans_annotating(args: argparse.Namespace) -> None:
    # Load caption data
    caption_df = load_caption_data(args)

    # Define tokenizer - we use bart tokenizer because it has start and end token
    en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    #nmt_model = EasyNMT('opus-mt')
    nmt_model = EasyNMT('mbart50_m2m')

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

    for idx in tqdm(range(len(train_df)), desc='Annotating with BackTranslation...'):
        # Get image_name, caption
        image_name = caption_df['image_name'][idx]
        gold_caption = caption_df['caption_text'][idx]

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
            data_dict_en['image_names'].append(image_name)
            data_dict_en['captions'].append(result_sentences[i])
            data_dict_en['caption_numbers'].append(i+1)
            data_dict_en['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict_en['all_captions'].append(result_sentences)

    save_name = 'train_BT_EN.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(data_dict_en, f)
        print(f"Saved {save_name} in {preprocessed_path}")

def back_translation(model, src: str, num: int=4) -> list: # list of str (backtranslated captions)
    #assert num == len(tmp_lang)

    mid_result = [model.translate(src, target_lang=each_tmp_lang) for each_tmp_lang in tmp_lang]
    result = [model.translate(each_mid_result, target_lang='en') for each_mid_result in mid_result]

    return result
