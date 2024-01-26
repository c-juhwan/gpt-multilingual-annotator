# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from google.cloud import translate
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path
from task.captioning.preprocessing import load_caption_data

PROJECT_ID = os.environ['GCP_PROJECT_ID'] # GCP Project ID

def googletrans_annotating(args: argparse.Namespace) -> None:

    # Define language code
    if args.annotation_mode == 'googletrans_lv':
        lang_code = 'lv'
        out_lang_code = 'LV'
        tokenizer = AutoTokenizer.from_pretrained('joelito/legal-latvian-roberta-base')

        with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
            loaded_data = pickle.load(f)
    elif args.annotation_mode == 'googletrans_et':
        lang_code = 'et'
        out_lang_code = 'ET'
        tokenizer = AutoTokenizer.from_pretrained('tartuNLP/EstBERT')

        with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
            loaded_data = pickle.load(f)
    elif args.annotation_mode == 'googletrans_fi':
        lang_code = 'fi'
        out_lang_code = 'FI'
        tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-uncased-v1')

        with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
            loaded_data = pickle.load(f)
    elif args.annotation_mode == 'googletrans_vie':
        lang_code = 'vi'
        out_lang_code = 'VIE'
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

        with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
            loaded_data = pickle.load(f)
    elif args.annotation_mode == 'googletrans_pl':
        lang_code = 'pl'
        out_lang_code = 'PL'
        tokenizer = AutoTokenizer.from_pretrained('sdadas/polish-bart-base')

        with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_FLICKR_EN.pkl'), 'rb') as f:
            loaded_data = pickle.load(f)

    save_data = {
        'image_names': [],
        'caption_numbers': [],
        'captions': [],
        'all_captions': [],
        'input_ids': [],
        'tokenizer': tokenizer,
    }

    # Save data as pickle file
    all_caption = []
    preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
    check_path(preprocessed_path)
    for idx in tqdm(range(len(loaded_data['image_names'])), desc='Annotating with Google Translator...'):
        # Get image_name, caption
        image_name = loaded_data['image_names'][idx]
        coco_eng_caption = loaded_data['captions'][idx]
        caption_number = loaded_data['caption_numbers'][idx]

        # Translate
        translated_caption = call_google_translate(coco_eng_caption, lang_code)

        if caption_number == 1:
            all_caption = [] # reset all_caption
        all_caption.append(translated_caption)

        # Tokenize translated caption
        tokenized_caption = tokenizer(translated_caption, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to save_data
        save_data['image_names'].append(image_name)
        save_data['caption_numbers'].append(caption_number)
        save_data['captions'].append(translated_caption)
        save_data['all_captions'].append(all_caption)
        save_data['input_ids'].append(tokenized_caption['input_ids'].squeeze())

    # Save data as pickle file
    with open(os.path.join(preprocessed_path, f'train_GOOGLETRANS_{out_lang_code}.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

def call_google_translate(input_text: str, target_lang_code: str) -> str:
    # Instantiates a client

    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{PROJECT_ID}/locations/{location}"

    # The text to translate
    patience = 0
    while True:
        try:
            response = client.translate_text(
                request={
                    "parent": parent,
                    "contents": [input_text],
                    "mime_type": "text/plain",  # mime types: text/plain, text/html
                    "source_language_code": "en-US",
                    "target_language_code": target_lang_code
                }
            )
        except Exception as e:
            print(e)
            time.sleep(1)
            patience += 1
            if patience > 10:
                print(f'Patience limit exceeded')
            else:
                continue # Try again if error occurs
        break

    for translation in response.translations:
        output = translation.translated_text

    return output
