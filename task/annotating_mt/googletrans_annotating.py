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

PROJECT_ID = os.environ['GCP_PROJECT_ID'] # GCP Project ID

def googletrans_annotating(args: argparse.Namespace) -> None:
    # Define language code
    if args.annotation_mode == 'googletrans_de':
        lang_code = 'de'
        tokenizer_lang_code = 'de_DE'
        out_lang_code = 'DE'
    if args.annotation_mode == 'googletrans_lv': # Latvian
        lang_code = 'lv'
        tokenizer_lang_code = 'lv_LV'
        out_lang_code = 'LV'
    if args.annotation_mode == 'googletrans_et': # Estonian
        lang_code = 'et'
        tokenizer_lang_code = 'et_EE'
        out_lang_code = 'ET'
    if args.annotation_mode == 'googletrans_fi': # Finnish
        lang_code = 'fi'
        tokenizer_lang_code = 'fi_FI'
        out_lang_code = 'FI'

    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=tokenizer_lang_code, tgt_lang=tokenizer_lang_code)
    with open(os.path.join(args.preprocess_path, 'machine_translation', args.task_dataset, 'train_ORIGINAL_DE.pkl'), 'rb') as f:
        loaded_data = pickle.load(f)

    save_data = {
        'source_text': [],
        'target_text': [],
        'text_number': [],
        'model_input_ids': [],
        'tokenizer': tokenizer
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, 'machine_translation', args.task_dataset)
    check_path(preprocessed_path)
    for idx in tqdm(range(len(loaded_data['source_text'][:6000])), desc=f'Annotating with Google Translator...'):
        # Get data
        source_text = loaded_data['source_text'][idx]
        text_number = loaded_data['text_number'][idx]

        # Translate
        translated_target_text = call_google_translate(source_text, lang_code)

        # Tokenize translated text
        tokenized = tokenizer(source_text, text_target=translated_target_text, padding='max_length', truncation=True,
                            max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to the data_dict
        save_data['source_text'].append(source_text)
        save_data['target_text'].append(translated_target_text)
        save_data['text_number'].append(text_number)
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
