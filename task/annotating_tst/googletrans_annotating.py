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
from task.text_style_transfer.preprocessing import get_dataset_path

PROJECT_ID = os.environ['GCP_PROJECT_ID'] # GCP Project ID

def googletrans_annotating(args: argparse.Namespace) -> None:
    # Define tokenizer
    _, tokenizer_lang_code = get_dataset_path(args)
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=tokenizer_lang_code, tgt_lang=tokenizer_lang_code)

    if tokenizer_lang_code == 'pt_XX':
        lang_code = 'pt'
        out_lang_code = 'PT'
    elif tokenizer_lang_code == 'fr_XX':
        lang_code = 'fr'
        out_lang_code = 'FR'
    elif tokenizer_lang_code == 'it_IT':
        lang_code = 'it'
        out_lang_code = 'IT'

    # Define data_dict
    with open(os.path.join(args.preprocess_path, 'text_style_transfer', 'gyafc_en', 'train_ORIGINAL_EN.pkl'), 'rb') as f:
        gyafc_data = pickle.load(f)

    save_data = {
        'informal_text': [],
        'formal_text': [],
        'all_references': [],
        'text_number': [],
        'category': [],
        'model_input_ids': [],
        'tokenizer': tokenizer,
    }

    # Save data as pickle file
    all_references = []
    preprocessed_path = os.path.join(args.preprocess_path, 'text_style_transfer', args.task_dataset)
    check_path(preprocessed_path)
    for idx in tqdm(range(len(gyafc_data['informal_text'])), desc='Annotating with Google Translator...'):
        # Get data
        informal_text = gyafc_data['informal_text'][idx]
        formal_text = gyafc_data['formal_text'][idx]
        text_number = gyafc_data['text_number'][idx]
        category = gyafc_data['category'][idx]

        # Translate each sentence
        translated_informal = call_google_translate(informal_text, lang_code)
        translated_formal = call_google_translate(formal_text, lang_code)

        if text_number == 1:
            all_references = []
        all_references.append(translated_formal)

        # Tokenize translated text
        tokenized = tokenizer(translated_informal, text_target=translated_formal, padding='max_length', truncation=True,
                              max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to the data_dict
        save_data['informal_text'].append(translated_informal)
        save_data['formal_text'].append(translated_formal)
        save_data['all_references'].append(all_references)
        save_data['text_number'].append(text_number)
        save_data['category'].append(category)
        save_data['model_input_ids'].append(tokenized['input_ids'].squeeze())

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