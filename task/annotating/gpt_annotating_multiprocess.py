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
    {"role": "system", "content": "You will generate paraphrases of the sentence and its translation in Korean language."},
    {"role": "system", "content": "VERY IMPORTANT: You must speak '-하다' form in Korean. You must not use '-합니다' or other forms. \
한국어 문장을 번역하여 생성할 때, 반드시 '-하다' 체를 사용하여야 한다. '-합니다', '-입니다' 등의 표현을 절대 사용하지 않는다."},
    {"role": "system", "content": "You will generate a translation of input sentence in Korean, and also generate 4 paraphrases and its translaton in Korean."},
    {"role": "system", "content": "Output sentence should be neutral expression. You should not generate phrases like 'You will see' or 'You will find'."},
    {"role": "system", "content": "Output sentence will be complete, natural and fluent."},
    {"role": "system", "content": "You will not generate the same sentence as the input sentence."},
    {"role": "system", "content": "You must not generate any biased, offensive, or inappropriate paraphrases."},
    {"role": "system", "content": "User input example: The men at bat readies to swing at the pitch while the umpire looks on.\n"},
    {"role": "system", "content": "Your output example: \n"},
    {"role": "system", "content": "Translation: 타석에 있는 남자들이 심판이 지켜보는 동안 스윙할 준비를 한다.\n\
Paraphrase 1: The male players at the bat ready to hit the ball as the umpire watches attentively. / 심판이 주의 깊게 지켜보는 가운데 배트를 든 남자 선수들이 공을 칠 준비를 하고 있다.\n\
Paraphrase 2: The male batters at the bat prepare to hit the pitch as the umpire stands watch. / 타석에 선 남성 타자들이 심판이 지켜보는 가운데 타구를 칠 준비를 하고 있다.\n\
Paraphrase 3: The batters at the plate are poised to swing as the umpire keeps an eye on them. / 타석에 있는 타자가 심판이 지켜보는 가운데 스윙할 자세를 취한다.\n\
Paraphrase 4: The hitters at the plate wait for themselves to take their swings at the ball while the umpire looks on. / 타석에 선 타자들은 심판이 지켜보는 동안 공을 향해 스윙할 준비를 한다.\n"},
    {"role": "system", "content": "You will not say 'Sure! here's the output' or any similar phrases."},
    {"role": "system", "content": "You will not say 'I don't know' or any similar phrases."},
    {"role": "system", "content": "You will just generate the output paraphrases following the output example."},
    {"role": "user", "content": "Input: Living room with furniture with garage door at one end."},
]

en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
ko_tokenizer = AutoTokenizer.from_pretrained('cosmoquester/bart-ko-base')
NUM_PROCESS = 8
tqdm_bar = tqdm(total=100, desc='Progress', position=0)

def gpt_annotating_multiprocess(args: argparse.Namespace) -> None:
    """
    Using multiprocessing
    """
    # Load caption data
    caption_df = load_caption_data(args)

    # Define data_dict
    data_dict_en = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }
    data_dict_ko = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': ko_tokenizer,
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
    check_path(preprocessed_path)

    # for split == 0, only remain caption_number == 1
    train_df = caption_df[caption_df['split'] == 0]
    # Remain only 1 caption per each image_name
    train_df = train_df.groupby('image_name').first().reset_index()
    train_df.reset_index(drop=True, inplace=True)
    print(train_df)

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
       results = p.starmap(try_call_gpt, starmap_items)

    print("Done with multiprocessing")

    for result in results:
        data_dict_en['image_names'] += result[0]['image_names'] # result[0] is data_dict_en
        data_dict_en['captions'] += result[0]['captions']
        data_dict_en['all_captions'] += result[0]['all_captions']
        data_dict_en['caption_numbers'] += result[0]['caption_numbers']
        #data_dict_en['input_ids'] += result[0]['input_ids'] # This will be done after concatenating all data_dict_en

        data_dict_ko['image_names'] += result[1]['image_names'] # result[1] is data_dict_ko
        data_dict_ko['captions'] += result[1]['captions']
        data_dict_ko['all_captions'] += result[1]['all_captions']
        data_dict_ko['caption_numbers'] += result[1]['caption_numbers']
        #data_dict_ko['input_ids'] += result[1]['input_ids'] # This will be done after concatenating all data_dict_ko

    for idx in tqdm(range(len(data_dict_en['captions'])), desc='Tokenizing English captions'):
        cap = data_dict_en['captions'][idx]
        tokenized = en_tokenizer(cap, padding='max_length', truncation=True,
                                 max_length=args.max_seq_len, return_tensors='pt')
        data_dict_en['input_ids'].append(tokenized['input_ids'].squeeze())
    for idx in tqdm(range(len(data_dict_ko['captions'])), desc='Tokenizing Korean captions'):
        cap = data_dict_ko['captions'][idx]

        ko_tokenized = ko_tokenizer(cap, padding='max_length', truncation=True,
                                    max_length=args.max_seq_len-1, return_tensors='pt') # -1 for [BOS]

        ko_tokenized_ = torch.cat([torch.tensor([ko_tokenizer.bos_token_id]), # ko_tokenizer requires manual [BOS] and [EOS]
                                    ko_tokenized['input_ids'].squeeze()], dim=0)
        # Change the first padding token to [EOS]
        first_pad_idx = torch.where(ko_tokenized_ == ko_tokenizer.pad_token_id)[0][0]
        ko_tokenized_[first_pad_idx] = ko_tokenizer.eos_token_id

        data_dict_ko['input_ids'].append(ko_tokenized_.squeeze())

    assert len(data_dict_en['image_names']) == len(data_dict_en['captions']) == len(data_dict_en['all_captions']) == len(data_dict_en['caption_numbers']) == len(data_dict_en['input_ids'])
    assert len(data_dict_ko['image_names']) == len(data_dict_ko['captions']) == len(data_dict_ko['all_captions']) == len(data_dict_ko['caption_numbers']) == len(data_dict_ko['input_ids'])

    # Save data_dict_en & data_dict_ko as pickle file
    if args.gpt_model_version == 'gpt-3.5-turbo':
        save_name_en = 'train_GPT35_EN.pkl'
        save_name_ko = 'train_GPT35_KO.pkl'
    elif args.gpt_model_version == 'gpt-4':
        save_name_en = 'train_GPT4_EN.pkl'
        save_name_ko = 'train_GPT4_KO.pkl'

    with open(os.path.join(preprocessed_path, save_name_en), 'wb') as f:
        pickle.dump(data_dict_en, f)
        print(f"Saved {save_name_en} in {preprocessed_path}")
    with open(os.path.join(preprocessed_path, save_name_ko), 'wb') as f:
        pickle.dump(data_dict_ko, f)
        print(f"Saved {save_name_ko} in {preprocessed_path}")

    tqdm_bar.close()

def try_call_gpt(args: argparse.Namespace, train_df_subset: pd.DataFrame) -> dict:
    try:
        return call_gpt(args, train_df_subset)
    except KeyboardInterrupt as k:
        raise k
    except Exception as e:
        logging.exception(f"Error in try_call_gpt: {e}")

def call_gpt(args: argparse.Namespace, train_df_subset: pd.DataFrame) -> dict:
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
    subset_dict_ko = {
        'image_names': [],
        'captions': [],
        'all_captions': [],
        'caption_numbers': [],
        'input_ids': [],
        'tokenizer': ko_tokenizer,
    }

    for idx in range(len(train_df_subset)):
        # Get image_name, caption
        image_name = train_df_subset['image_name'][idx]
        gold_caption = train_df_subset['caption_text'][idx]

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
                result_sentences.append({"en": gold_caption, "ko": gpt_sentences[0]})
                for i in range(1, len(gpt_sentences)):
                    result_sentences.append({"en": gpt_sentences[i].split(" / ")[0], "ko": gpt_sentences[i].split(" / ")[1]})
            except KeyboardInterrupt as k:
                raise k # if KeyboardInterrupt, raise it to stop the program
            except:
                # if gpt_response is not correctly generated, print error message and try again
                #print(f"Error {error_counter}: {gpt_response}")
                error_counter += 1
                if error_counter >= 3:
                    #print("Error: Too many errors. Skip this image.")
                    break
                continue

            if len(result_sentences) == 5: # if gpt_response is correctly generated and has 5 sentences
                break # break the while loop
            else:
                # if gpt_response is not correctly generated, print error message and try again
                error_counter += 1
                if error_counter >= 3:
                    #print("Error: Too many errors. Skip this image.")
                    break
                #print(f"Error {error_counter}: {gpt_response}")
                continue
        if error_counter >= 3:
            continue # skip this image

        # Tokenize and append to data_dict_en & data_dict_ko
        for i in range(len(result_sentences)):
            # Tokenize
            #en_tokenized = en_tokenizer(result_sentences[i]['en'], padding='max_length', truncation=True,
            #                            max_length=args.max_seq_len, return_tensors='pt')
            #ko_tokenized = ko_tokenizer(result_sentences[i]['ko'], padding='max_length', truncation=True,
            #                            max_length=args.max_seq_len, return_tensors='pt')

            # Append to data_dict_en
            subset_dict_en['image_names'].append(image_name)
            subset_dict_en['captions'].append(result_sentences[i]['en'])
            subset_dict_en['caption_numbers'].append(i+1)
            #subset_dict_en['input_ids'].append(en_tokenized['input_ids'].squeeze()) # This will be done after multiprocessing
            subset_dict_en['all_captions'].append([result_sentences[i]['en'] for i in range(len(result_sentences))])

            # Append to data_dict_ko
            subset_dict_ko['image_names'].append(image_name)
            subset_dict_ko['captions'].append(result_sentences[i]['ko'])
            subset_dict_ko['caption_numbers'].append(i+1)
            #subset_dict_ko['input_ids'].append(ko_tokenized['input_ids'].squeeze()) # This will be done after multiprocessing
            subset_dict_ko['all_captions'].append([result_sentences[i]['ko'] for i in range(len(result_sentences))])

        tqdm_bar.update(1)

    return subset_dict_en, subset_dict_ko # return data_dict_en & data_dict_ko
