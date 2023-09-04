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
from task.text_style_transfer.preprocessing import get_dataset_path

prompt_message = {
    'FR': [
        {"role": "system", "content": "You are a helpful assistant. You are fluent in French and English."},
        {"role": "system", "content": "You will generate paraphrases of formal and informal sentences and their translations into French."},
        {"role": "system", "content": "Output sentence should be neutral expression."},
        {"role": "system", "content": "Output sentence will be complete, natural and fluent."},
        {"role": "system", "content": "Each output sentence should have different expressions as much as possible."},
        {"role": "system", "content": "You will not generate the same sentence as the input sentence."},
        {"role": "system", "content": "You must not generate any biased, offensive, or inappropriate paraphrases."},
        {"role": "system", "content": "You will not say 'Sure! here's the output' or any similar phrases."},
        {"role": "system", "content": "You will not say 'I don't know' or any similar phrases."},
        {"role": "system", "content": "You will just generate the output paraphrases following the output example."},
        {"role": "user",   "content": "[Input Sentence]\n\
Formal 1: Then kiss her, brother; that works every time.\n\
Informal 1: Then kiss her;) works every time bro!!!!"},
        {"role": "assistant", "content": "[Paraphrase]\n\
Formal 2: Subsequently, kiss her, sibling; that method proves effective on each occasion.\n\
Informal 2: So, just give her a smooch, bro! It seriously works every single time ;)\n\n\
[Translation in French]\n\
Formal 1: Alors embrasse-la, mon frère. Cela fonctionne à chaque fois.\n\
Informal 1: Alors embrasse-la ;) ça marche à chaque fois frérot!!!!\n\
Formal 2: Ensuite, embrasse-la, frère ; cette méthode fonctionne à chaque fois.\n\
Informal 2: Alors, donne-lui un bisou, mec ! Ça marche à tous les coups ;)"},
        {"role": "user",   "content": "[Input Sentence]\n\
Formal 1: After that I never bought her another gift.\n\
Informal 1: and enver since then i never bought her another gift"},
    ],
    'PT': [
        {"role": "system", "content": "You are a helpful assistant. You are fluent in Brazilian Portuguese and English."},
        {"role": "system", "content": "You will generate paraphrases of formal and informal sentences and their translations into Brazilian Portuguese."},
        {"role": "system", "content": "Output sentence should be neutral expression."},
        {"role": "system", "content": "Output sentence will be complete, natural and fluent."},
        {"role": "system", "content": "Each output sentence should have different expressions as much as possible."},
        {"role": "system", "content": "You will not generate the same sentence as the input sentence."},
        {"role": "system", "content": "You must not generate any biased, offensive, or inappropriate paraphrases."},
        {"role": "system", "content": "You will not say 'Sure! here's the output' or any similar phrases."},
        {"role": "system", "content": "You will not say 'I don't know' or any similar phrases."},
        {"role": "system", "content": "You will just generate the output paraphrases following the output example."},
        {"role": "user",   "content": "[Input Sentence]\n\
Formal 1: Then kiss her, brother; that works every time.\n\
Informal 1: Then kiss her;) works every time bro!!!!"},
        {"role": "assistant", "content": "[Paraphrase]\n\
Formal 2: Subsequently, kiss her, sibling; that method proves effective on each occasion.\n\
Informal 2: So, just give her a smooch, bro! It seriously works every single time ;)\n\n\
[Translation in Brazilian Portuguese]\n\
Formal 1: Então beije-a, irmão. Isso funciona toda vez.\n\
Informal 1: Aí beija ela ;) sempre funciona, mano!!!!\n\
Formal 2: Em seguida, abrace-a, irmão; esse método funciona toda vez.\n\
Informal 2: Então, só dá um beijo nela, bro! Funciona sério toda vez ;)"},
        {"role": "user",   "content": "[Input Sentence]\n\
Formal 1: After that I never bought her another gift.\n\
Informal 1: and enver since then i never bought her another gift"},
    ],
    'IT': [
        {"role": "system", "content": "You are a helpful assistant. You are fluent in Italian and English."},
        {"role": "system", "content": "You will generate paraphrases of formal and informal sentences and their translations into Italian."},
        {"role": "system", "content": "Output sentence should be neutral expression."},
        {"role": "system", "content": "Output sentence will be complete, natural and fluent."},
        {"role": "system", "content": "Each output sentence should have different expressions as much as possible."},
        {"role": "system", "content": "You will not generate the same sentence as the input sentence."},
        {"role": "system", "content": "You must not generate any biased, offensive, or inappropriate paraphrases."},
        {"role": "system", "content": "You will not say 'Sure! here's the output' or any similar phrases."},
        {"role": "system", "content": "You will not say 'I don't know' or any similar phrases."},
        {"role": "system", "content": "You will just generate the output paraphrases following the output example."},
        {"role": "user",   "content": "[Input Sentence]\n\
Formal 1: Then kiss her, brother; that works every time.\n\
Informal 1: Then kiss her;) works every time bro!!!!"},
        {"role": "assistant", "content": "[Paraphrase]\n\
Formal 2: Subsequently, kiss her, sibling; that method proves effective on each occasion.\n\
Informal 2: So, just give her a smooch, bro! It seriously works every single time ;)\n\n\
[Translation in Italian]\n\
Formal 1: Allora baciala, fratello, funziona ogni volta.\n\
Informal 1: Poi baciala;) funziona ogni volta fratello!!!!\n\
Formal 2: Quindi, baciala, fratello; questo metodo si dimostra efficace ogni volta.\n\
Informal 2: Allora, daile un bacio, bro! Funziona davvero ogni singola volta ;)"},
        {"role": "user",   "content": "[Input Sentence]\n\
Formal 1: After that I never bought her another gift.\n\
Informal 1: and enver since then i never bought her another gift"},
    ]
}

NUM_PROCESS = 8
tqdm_bar = tqdm(total=3000, desc='Progress', position=0)

def gpt_annotating_multiprocess(args: argparse.Namespace) -> None:
    # Load dataset
    _, lang_code = get_dataset_path(args)
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=lang_code, tgt_lang=lang_code)

    if lang_code == 'pt_XX':
        out_lang_code = 'PT'
    elif lang_code == 'fr_XX':
        out_lang_code = 'FR'
    elif lang_code == 'it_IT':
        out_lang_code = 'IT'
    args.out_lang_code = out_lang_code

    # Define data_dict
    with open(os.path.join(args.preprocess_path, 'text_style_transfer', 'gyafc_en', 'train_ORIGINAL_EN.pkl'), 'rb') as f:
        gyafc_data = pickle.load(f)

    # We will use only half of the data
    gyafc_data['informal_text'] = gyafc_data['informal_text'][:len(gyafc_data['informal_text'])//2]
    gyafc_data['formal_text'] = gyafc_data['formal_text'][:len(gyafc_data['formal_text'])//2]
    gyafc_data['all_references'] = gyafc_data['all_references'][:len(gyafc_data['all_references'])//2]
    gyafc_data['text_number'] = gyafc_data['text_number'][:len(gyafc_data['text_number'])//2]
    gyafc_data['category'] = gyafc_data['category'][:len(gyafc_data['category'])//2]
    gyafc_data['model_input_ids'] = gyafc_data['model_input_ids'][:len(gyafc_data['model_input_ids'])//2]

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
    preprocessed_path = os.path.join(args.preprocess_path, 'text_style_transfer', args.task_dataset)
    check_path(preprocessed_path)

    # Call multiprocessing using starmap
    # Divide train_df into NUM_PROCESS parts
    informal_text_divided_list = []
    formal_text_divided_list = []
    for i in range(NUM_PROCESS):
        informal_text_divided_list.append(gyafc_data['informal_text'][i::NUM_PROCESS])
        formal_text_divided_list.append(gyafc_data['formal_text'][i::NUM_PROCESS])
    tqdm_bar.total = len(gyafc_data['informal_text'])

    # Call multiprocessing
    starmap_items = [
        (args, informal_text_divided_list[i], formal_text_divided_list[i]) for i in range(NUM_PROCESS)
    ]

    print(f"Start multiprocessing with {NUM_PROCESS} processes")
    print(f"Total {len(gyafc_data['informal_text'])} data will be processed")

    with Pool(NUM_PROCESS) as p:
        results = p.starmap(try_call_gpt, starmap_items)

    print("Done with multiprocessing")
    tqdm_bar.close()
    with open(os.path.join(preprocessed_path, 'GPT_TEMP.pkl'), 'wb') as f:
        pickle.dump(results, f)

    for result in results:
        save_data['informal_text'] += result['informal_text']
        save_data['formal_text'] += result['formal_text']
        save_data['all_references'] += result['all_references'] # We don't use this here - so add None
        save_data['text_number'] += result['text_number']
        save_data['category'] += result['category']
        save_data['model_input_ids'] += result['model_input_ids'] # Currently, we don't use model_input_ids

    for idx in tqdm(range(len(save_data['informal_text'])), desc='Tokenizing...'):
        tokenized = tokenizer(save_data['informal_text'][idx], text_target=save_data['formal_text'][idx],
                              padding='max_length', truncation=True, max_length=100, return_tensors='pt')

        save_data['model_input_ids'][idx] = tokenized['input_ids'].squeeze()

    # Save data as pickle file
    assert args.gpt_model_version == 'gpt-4' # we will only use gpt-4 here
    with open(os.path.join(preprocessed_path, f'train_GPT4_{out_lang_code}.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
        print(f"Saved train_GPT4_{out_lang_code}.pkl in {preprocessed_path}")

def try_call_gpt(args: argparse.Namespace, informal_text_sublist: list, formal_text_sublist: list) -> dict:
    assert len(informal_text_sublist) == len(formal_text_sublist), "Length of informal_text_sublist and formal_text_sublist must be same"

    try:
        return call_gpt(args, informal_text_sublist, formal_text_sublist)
    except KeyboardInterrupt as k:
        raise k
    except Exception as e:
        logging.exception(f"Error in try_call_gpt: {e}")

def call_gpt(args: argparse.Namespace, informal_text_sublist: list, formal_text_sublist: list) -> dict:
    """
    Call GPT API
    """
    subset_dict = {
        'informal_text': [],
        'formal_text': [],
        'all_references': [],
        'text_number': [],
        'category': [],
        'model_input_ids': [],
        'tokenizer': None,
    }

    for idx in range(len(informal_text_sublist)):
        # Get data
        informal_text = informal_text_sublist[idx]
        formal_text = formal_text_sublist[idx]

        # Remove last user input from prompt_message and add new user input
        prompt_message[args.out_lang_code].pop()
        prompt_message[args.out_lang_code].append({"role": "user", "content": f"[Input Sentence]\n\
Formal 1: {formal_text}\n\
Informal 1: {informal_text}"})

        error_counter = 0
        while True:
            try:
                # Get gpt response
                gpt_response = openai.ChatCompletion.create(
                    model=args.gpt_model_version,
                    messages=prompt_message[args.out_lang_code],
                )

                gpt_sentences = gpt_response['choices'][0]['message']['content']

                # Break down response into two part
                paraphrases = gpt_sentences.split('\n\n')[0]
                translations = gpt_sentences.split('\n\n')[1]

                # We only use translations
                translations = translations.split('\n')
                if len(translations) != 5:
                    # if translations is not correctly generated, print error message and try again
                    raise ValueError("Error: translations is not correctly generated")

                formal_1 = translations[1].split('Formal 1: ')[1]
                informal_1 = translations[2].split('Informal 1: ')[1]
                formal_2 = translations[3].split('Formal 2: ')[1]
                informal_2 = translations[4].split('Informal 2: ')[1]

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
        subset_dict['informal_text'].extend([informal_1, informal_2])
        subset_dict['formal_text'].extend([formal_1, formal_2])
        subset_dict['all_references'].extend([None, None])
        subset_dict['text_number'].extend([1, 2])
        subset_dict['category'].extend(['fr', 'fr'])
        subset_dict['model_input_ids'].extend([None, None])

        if tqdm_bar.n + NUM_PROCESS <= tqdm_bar.total:
           tqdm_bar.update(NUM_PROCESS)

    return subset_dict
