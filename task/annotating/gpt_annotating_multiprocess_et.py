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
from task.captioning.preprocessing import load_caption_data

prompt_message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "system", "content": "User will ask you to generate paraphrases of a sentence."},
    {"role": "system", "content": "You will generate paraphrases of the sentence and its translation in Estonian language."},
    {"role": "system", "content": "You will generate a translation of input sentence in Estonian, and also generate 4 paraphrases and its translaton in Estonian."},
    {"role": "system", "content": "Output sentence should be neutral expression. You should not generate phrases like 'You will see' or 'You will find'."},
    {"role": "system", "content": "Output sentence will be complete, natural and fluent."},
    {"role": "system", "content": "Each output sentence should have different expressions as much as possible."},
    {"role": "system", "content": "You will not generate the same sentence as the input sentence."},
    {"role": "system", "content": "You must not generate any biased, offensive, or inappropriate paraphrases."},
    {"role": "system", "content": "User input example: The men at bat readies to swing at the pitch while the umpire looks on.\n"},
    {"role": "system", "content": "Your output example: \n"},
    {"role": "system", "content": "Translation: Kurikamehed on valmis väljakul kõikuma, samal ajal kui kohtunik pealt vaatab.\n\
Paraphrase 1: The male players at the bat ready to hit the ball as the umpire watches attentively. / Kurika juures olevad meesmängijad on valmis palli lööma, kui kohtunik jälgib tähelepanelikult.\n\
Paraphrase 2: The male batters at the bat prepare to hit the pitch as the umpire stands watch. / Kurikameeste lööjad valmistuvad väljakule lööma, samal ajal kui kohtunik seisab.\n\
Paraphrase 3: The batters at the plate are poised to swing as the umpire keeps an eye on them. / Taldrikul olevad lööjad on valmis kõikuma, kui kohtunik hoiab neil silma peal.\n\
Paraphrase 4: The hitters at the plate wait for themselves to take their swings at the ball while the umpire looks on. / Taldrikul olevad lööjad ootavad, kuni kohtunik vaatab pealt, et palli juures hoo sisse lüüa.\n"},
    {"role": "system", "content": "You will not say 'Sure! here's the output' or any similar phrases."},
    {"role": "system", "content": "You will not say 'I don't know' or any similar phrases."},
    {"role": "system", "content": "You will just generate the output paraphrases following the output example."},
    {"role": "user", "content": "Input: Living room with furniture with garage door at one end."},
]
en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
et_tokenizer = AutoTokenizer.from_pretrained('tartuNLP/EstBERT')
NUM_PROCESS = 8
tqdm_bar = tqdm(total=100, desc='Progress', position=0)

def gpt_annotating_multiprocess_et(args: argparse.Namespace) -> None:
    # Define data_dict
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
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
    total_length = len(loaded_data['image_names'])
    print(f"Total {total_length} data points")
    for idx in range(total_length):
        if loaded_data['caption_numbers'][idx] == 1:
            train_data_dict['image_names'].append(loaded_data['image_names'][idx])
            train_data_dict['caption_numbers'].append(loaded_data['caption_numbers'][idx])
            train_data_dict['captions'].append(loaded_data['captions'][idx])
            train_data_dict['all_captions'].append(loaded_data['all_captions'][idx])
            train_data_dict['input_ids'].append(loaded_data['input_ids'][idx])
        else:
            continue
    assert len(train_data_dict['image_names']) == len(train_data_dict['caption_numbers']) == len(train_data_dict['captions']) == len(train_data_dict['all_captions']) == len(train_data_dict['input_ids']), f"train_data_dict lengths are not equal"
    #assert len(train_data_dict['image_names']) == total_length // 5, f"len(train_data_dict['image_names']):{len(train_data_dict['image_names'])} != total_length // 5:{total_length // 5}"
    if len(train_data_dict['image_names']) != total_length // 5:
        #raise UserWarning(f"len(train_data_dict['image_names']):{len(train_data_dict['image_names'])} != total_length // 5:{total_length // 5}")
        print(f"len(train_data_dict['image_names']):{len(train_data_dict['image_names'])} != total_length // 5:{total_length // 5}")

    # Define data_dict
    data_dict_en = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }
    data_dict_et = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': et_tokenizer,
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
    check_path(preprocessed_path)

    # Call multiprocessing using starmap
    # Divide train_df into NUM_PROCESS parts
    image_names_divided_list = []
    captions_divided_list = []
    for i in range(NUM_PROCESS):
        image_names_divided_list.append(train_data_dict['image_names'][i::NUM_PROCESS])
        captions_divided_list.append(train_data_dict['captions'][i::NUM_PROCESS])
    tqdm_bar.total = len(train_data_dict['image_names'])

    # Call multiprocessing
    starmap_items = [
        (args, image_names_divided_list[i], captions_divided_list[i]) for i in range(NUM_PROCESS)
    ]

    print(f"Start multiprocessing with {NUM_PROCESS} processes")
    print(f"Total {len(train_data_dict['image_names'])} individual image-caption pairs will be processed")

    with Pool(NUM_PROCESS) as p:
       results = p.starmap(try_call_gpt, starmap_items)

    print("Done with multiprocessing")
    tqdm_bar.close()
    with open(os.path.join(preprocessed_path, 'GPT_TEMP.pkl'), 'wb') as f:
        pickle.dump(results, f)

    for result in results:
        data_dict_en['image_names'] += result[0]['image_names'] # result[0] is data_dict_en
        data_dict_en['captions'] += result[0]['captions']
        data_dict_en['all_captions'] += result[0]['all_captions']
        data_dict_en['caption_numbers'] += result[0]['caption_numbers']

        data_dict_et['image_names'] += result[1]['image_names'] # result[1] is data_dict_et
        data_dict_et['captions'] += result[1]['captions']
        data_dict_et['all_captions'] += result[1]['all_captions']
        data_dict_et['caption_numbers'] += result[1]['caption_numbers']

    for idx in tqdm(range(len(data_dict_en['captions'])), desc='Tokenizing English captions'):
        cap = data_dict_en['captions'][idx]
        tokenized = en_tokenizer(cap, padding='max_length', truncation=True,
                                 max_length=args.max_seq_len, return_tensors='pt')
        data_dict_en['input_ids'].append(tokenized['input_ids'].squeeze())
    for idx in tqdm(range(len(data_dict_et['captions'])), desc='Tokenizing Estonian captions'):
        cap = data_dict_et['captions'][idx]
        tokenized = et_tokenizer(cap, padding='max_length', truncation=True,
                                  max_length=args.max_seq_len, return_tensors='pt')
        data_dict_et['input_ids'].append(tokenized['input_ids'].squeeze())

    assert len(data_dict_en['image_names']) == len(data_dict_en['captions']) == len(data_dict_en['all_captions']) == len(data_dict_en['caption_numbers']) == len(data_dict_en['input_ids']), f"data_dict_en lengths are not equal"
    assert len(data_dict_en['image_names']) == len(data_dict_et['image_names']), f"len(data_dict_en['image_names']):{len(data_dict_en['image_names'])} != len(data_dict_et['image_names']):{len(data_dict_et['image_names'])}"
    assert len(data_dict_et['image_names']) == len(data_dict_et['captions']) == len(data_dict_et['all_captions']) == len(data_dict_et['caption_numbers']) == len(data_dict_et['input_ids']), f"data_dict_et lengths are not equal"

    # Save data_dict_en & data_dict_et as pickle file
    if args.gpt_model_version == 'gpt-3.5-turbo':
        save_name_en = 'train_GPT35_EN.pkl'
        save_name_et = 'train_GPT35_ET.pkl'
    elif args.gpt_model_version == 'gpt-4':
        save_name_en = 'train_GPT4_EN.pkl'
        save_name_et = 'train_GPT4_ET.pkl'

    with open(os.path.join(preprocessed_path, save_name_en), 'wb') as f:
        pickle.dump(data_dict_en, f)
        print(f"Saved {save_name_en} in {preprocessed_path}")
    with open(os.path.join(preprocessed_path, save_name_et), 'wb') as f:
        pickle.dump(data_dict_et, f)
        print(f"Saved {save_name_et} in {preprocessed_path}")

def try_call_gpt(args: argparse.Namespace, image_names_sublist: list, captions_sublist: list) -> dict:
    assert len(image_names_sublist) == len(captions_sublist), "image_names_sublist and captions_sublist must have the same length"

    try:
        return call_gpt(args, image_names_sublist, captions_sublist)
    except KeyboardInterrupt as k:
        raise k
    except Exception as e:
        logging.exception(f"Error in try_call_gpt: {e}")

def call_gpt(args: argparse.Namespace, image_names_sublist: list, captions_sublist: list) -> dict:
    """
    Call GPT-3 API
    """
    subset_dict_en = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }
    subset_dict_et = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': et_tokenizer,
    }

    for idx in range(len(image_names_sublist)):
        # Get image_name, caption
        image_name = image_names_sublist[idx]
        gold_caption = captions_sublist[idx]

        # Remove last user input from prompt_message and add new user input
        prompt_message.pop()
        prompt_message.append({"role": "user", "content": f"Input: {gold_caption}"})

        error_counter = 0
        while True:
            try:
                # Get gpt paraphrases
                gpt_response = openai.ChatCompletion.create(
                    model=args.gpt_model_version,
                    messages=prompt_message,
                )

                # Break down the response into sentences
                gpt_sentences = gpt_response['choices'][0]['message']['content'].split("\n")
                # Remove the ~: part
                for i in range(len(gpt_sentences)):
                    # Remove multiple spaces
                    gpt_sentences[i] = " ".join(gpt_sentences[i].split())
                    # Remove the ~: part
                    gpt_sentences[i] = gpt_sentences[i][gpt_sentences[i].find(":") + 2:]
                # Remove empty strings
                gpt_sentences = list(filter(None, gpt_sentences))

                result_sentences = []
                result_sentences.append({"en": gold_caption, "et": gpt_sentences[0]})
                for i in range(1, len(gpt_sentences)):
                    result_sentences.append({"en": gpt_sentences[i].split(" / ")[0], "et": gpt_sentences[i].split(" / ")[1]})
            except KeyboardInterrupt as k:
                raise k # if KeyboardInterrupt, raise it to stop the program
            except:
                # if gpt_response is not correctly generated, print error message and try again
                #print(f"Error {error_counter}: {gpt_response}")
                error_counter += 1
                if error_counter >= 3:
                    print("Error: Too many errors. Skip this image.")
                    break
                continue

            if len(result_sentences) == 5: # if gpt_response is correctly generated and has 5 sentences
                break # break the while loop
            else:
                # if gpt_response is not correctly generated, print error message and try again
                error_counter += 1
                if error_counter >= 3:
                    print("Error: Too many errors. Skip this image.")
                    break
                #print(f"Error {error_counter}: {gpt_response}")
                continue
        if error_counter >= 3:
            continue # skip this image

        # Append to data_dict_en & data_dict_et
        for i in range(len(result_sentences)):
            # Append to data_dict_en
            subset_dict_en['image_names'].append(image_name)
            subset_dict_en['captions'].append(result_sentences[i]['en'])
            subset_dict_en['caption_numbers'].append(i+1)
            subset_dict_en['all_captions'].append([result_sentences[i]['en'] for i in range(len(result_sentences))])

            # Append to data_dict_et
            subset_dict_et['image_names'].append(image_name)
            subset_dict_et['captions'].append(result_sentences[i]['et'])
            subset_dict_et['caption_numbers'].append(i+1)
            subset_dict_et['all_captions'].append([result_sentences[i]['et'] for i in range(len(result_sentences))])

        if tqdm_bar.n + NUM_PROCESS <= tqdm_bar.total:
            tqdm_bar.update(NUM_PROCESS)

    return subset_dict_en, subset_dict_et # return data_dict_en & data_dict_et
