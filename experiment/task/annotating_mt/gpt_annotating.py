# Standard Library Modules
import os
import sys
import time
import pickle
import logging
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
import multiprocessing as mp
from multiprocessing import Pool
# 3rd-party Modules
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

prompt_message = {
    'DE': [
        {"role": "system", "content": "You are a helpful assistant. You are fluent in German and English."},
        {"role": "system", "content": "You will generate paraphrases of given sentences and their translations into German."},
        {"role": "system", "content": "Output sentence should be neutral expression."},
        {"role": "system", "content": "Output sentence will be complete, natural and fluent."},
        {"role": "system", "content": "Each output sentence should have different expressions as much as possible."},
        {"role": "system", "content": "You will not generate the same sentence as the input sentence."},
        {"role": "system", "content": "You must not generate any biased, offensive, or inappropriate paraphrases."},
        {"role": "system", "content": "You will not say 'Sure! here's the output' or any similar phrases."},
        {"role": "system", "content": "You will not say 'I don't know' or any similar phrases."},
        {"role": "system", "content": "You will just generate the output paraphrases following the output example."},
        {"role": "user",   "content": "[Input Sentence]\n\
English 1: Two men pose while having a photograph taken."},
        {"role": "assistant", "content": "[Output Sentence]\n\
German 1: Zwei Männer posieren beim Fotografieren.\n\
English 2: A pair of males post for a picture being captured.\n\
German 2: Ein Duo von Männern posiert für ein aufgenommenes Bild."},
        {"role": "user",   "content": "[Input Sentence]\n\
English 1: A man taking a swing at a ball on the court.\n"},
    ],
}

NUM_PROCESS = 8
tqdm_bar = tqdm(total=3000, desc='Progress', position=0)

def gpt_annotating_multiprocess(args: argparse.Namespace) -> None:
    # Load dataset
    if args.annotation_mode in ['original_de', 'translated_de', 'gpt_de', 'googletrans_de']:
        lang_code = 'de_DE'
        out_lang_code = 'DE'
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=lang_code, tgt_lang=lang_code)
    args.out_lang_code = out_lang_code

    # Define data_dict
    with open(os.path.join(args.preprocess_path, 'machine_translation', 'multi30k', 'train_ORIGINAL_DE.pkl'), 'rb') as f:
        original_data = pickle.load(f)

    # We will use only half of the data
    original_data['source_text'] = original_data['source_text'][:len(original_data['source_text'])//2]
    original_data['text_number'] = original_data['text_number'][:len(original_data['text_number'])//2]
    original_data['model_input_ids'] = original_data['model_input_ids'][:len(original_data['model_input_ids'])//2]

    save_data = {
        'source_text': [],
        'target_text': [],
        'text_number': [],
        'model_input_ids': [],
        'tokenizer': tokenizer,
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, 'machine_translation', args.task_dataset)
    check_path(preprocessed_path)

    # Call multiprocessing using starmap
    # Divide train_df into NUM_PROCESS parts
    source_text_divided_list = []
    for i in range(NUM_PROCESS):
        source_text_divided_list.append(original_data['source_text'][i::NUM_PROCESS])
    tqdm_bar.total = len(original_data['source_text'])

    # Call multiprocessing
    starmap_items = [
        (args, source_text_divided_list[i]) for i in range(NUM_PROCESS)
    ]

    print(f"Start multiprocessing with {NUM_PROCESS} processes")
    print(f"Total {len(original_data['source_text'])} data will be processed")

    with Pool(NUM_PROCESS) as p:
        results = p.starmap(try_call_gpt, starmap_items)

    print("Done with multiprocessing")
    tqdm_bar.close()
    with open(os.path.join(preprocessed_path, 'GPT_TEMP.pkl'), 'wb') as f:
        pickle.dump(results, f)

    for result in results:
        save_data['source_text'] += result['source_text']
        save_data['target_text'] += result['target_text']
        save_data['all_references'] += result['all_references'] # We don't use this here - so add None
        save_data['text_number'] += result['text_number']
        save_data['category'] += result['category']
        save_data['model_input_ids'] += result['model_input_ids'] # Currently, we don't use model_input_ids

    for idx in tqdm(range(len(save_data['source_text'])), desc='Tokenizing...'):
        tokenized = tokenizer(save_data['source_text'][idx], text_target=save_data['target_text'][idx],
                              padding='max_length', truncation=True, max_length=100, return_tensors='pt')

        save_data['model_input_ids'][idx] = tokenized['input_ids'].squeeze()

    # Save data as pickle file
    assert args.gpt_model_version == 'gpt-4' # we will only use gpt-4 here
    with open(os.path.join(preprocessed_path, f'train_GPT4_{out_lang_code}.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
        print(f"Saved train_GPT4_{out_lang_code}.pkl in {preprocessed_path}")

def try_call_gpt(args: argparse.Namespace, source_text_sublist: list) -> dict:
    try:
        return call_gpt(args, source_text_sublist)
    except KeyboardInterrupt as k:
        raise k
    except Exception as e:
        logging.exception(f"Error in try_call_gpt: {e}")

def call_gpt(args: argparse.Namespace, source_text_sublist: list) -> dict:
    """
    Call GPT API
    """
    subset_dict = {
        'source_text': [],
        'target_text': [],
        'all_references': [],
        'text_number': [],
        'category': [],
        'model_input_ids': [],
        'tokenizer': None,
    }

    for idx in range(len(source_text_sublist)):
        # Get data
        source_text = source_text_sublist[idx]

        # Remove last user input from prompt_message and add new user input
        prompt_message[args.out_lang_code].pop()
        prompt_message[args.out_lang_code].append({"role": "user", "content": f"[Input Sentence]\n\
English 1: {source_text}\n"})

        error_counter = 0
        while True:
            try:
                # Get gpt response
                gpt_response = openai.ChatCompletion.create(
                    model=args.gpt_model_version,
                    messages=prompt_message[args.out_lang_code],
                )

                gpt_sentences = gpt_response['choices'][0]['message']['content']

                # Break down the response into sentences
                gpt_sentences = gpt_response['choices'][0]['message']['content'].split("\n")
                if len(gpt_sentences) != 4: # if gpt_response is not correctly generated, print error message and try again
                    raise ValueError("Error: output is not correctly generated")
                gpt_sentences = gpt_sentences[1:] # remove first line [Output Sentence]

                # Remove the ~: part
                for i in range(len(gpt_sentences)):
                    # Remove multiple spaces
                    gpt_sentences[i] = " ".join(gpt_sentences[i].split())
                    # Remove the ~: part
                    gpt_sentences[i] = gpt_sentences[i][gpt_sentences[i].find(":") + 2:]
                # Remove empty strings
                gpt_sentences = list(filter(None, gpt_sentences))

                english_1 = source_text
                target_1 = gpt_sentences[0]
                english_2 = gpt_sentences[1]
                target_2 = gpt_sentences[2]

                break # if gpt_response is correctly generated, break the while loop
            except:
                # print the category of the error
                # print(f"Error {error_counter}: {sys.exc_info()[0]}")
                # if gpt_response is not correctly generated, print error message and try again
                # print(f"Error {error_counter}: {gpt_response['choices'][0]['message']['content']}")
                error_counter += 1
                if error_counter >= 5:
                    print("Error: Too many errors. Skip this data.")
                    break
                continue

        # Append the data to the subset_dict
        subset_dict['source_text'].extend([english_1, english_2])
        subset_dict['target_text'].extend([target_1, target_2])
        subset_dict['text_number'].extend([1, 2])
        subset_dict['model_input_ids'].extend([None, None])

        if tqdm_bar.n + NUM_PROCESS <= tqdm_bar.total:
           tqdm_bar.update(NUM_PROCESS)

    return subset_dict
