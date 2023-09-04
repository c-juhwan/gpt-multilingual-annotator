# Standard Library Modules
import os
import sys
import time
import pickle
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
# 3rd-party Modules
from tqdm.auto import tqdm
# Huggingface Modules
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path
from task.captioning.preprocessing import load_caption_data

def translation_annotating_lv(args: argparse.Namespace) -> None:
    # Define tokenizer
    trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(args.device)
    trans2_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="lv_LV")
    trans2_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(args.device)
    lv_tokenizer = AutoTokenizer.from_pretrained('joelito/legal-latvian-roberta-base')

    # Define data_dict
    loaded_data = {}
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
        loaded_data['train'] = pickle.load(f)
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'valid_COCO_EN.pkl'), 'rb') as f:
        loaded_data['valid'] = pickle.load(f)
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'test_COCO_EN.pkl'), 'rb') as f:
        loaded_data['test'] = pickle.load(f)


    save_data = {
        'train': {
            'image_names': [],
            'caption_numbers': [],
            'captions': [],
            'all_captions': [],
            'input_ids': [],
            'tokenizer': lv_tokenizer,
        },
        'valid': {
            'image_names': [],
            'caption_numbers': [],
            'captions': [],
            'all_captions': [],
            'input_ids': [],
            'tokenizer': lv_tokenizer,
        },
        'test': {
            'image_names': [],
            'caption_numbers': [],
            'captions': [],
            'all_captions': [],
            'input_ids': [],
            'tokenizer': lv_tokenizer,
        }
    }

    for split in ['train']: # Train data will be annotated with NLLB
        # Save data as pickle file
        all_caption = []
        preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
        check_path(preprocessed_path)
        for idx in tqdm(range(len(loaded_data[split]['image_names'])), desc='Annotating Train Data with Translator...'):
            # Get image_name, caption
            image_name = loaded_data[split]['image_names'][idx]
            coco_eng_caption = loaded_data[split]['captions'][idx]
            caption_number = loaded_data[split]['caption_numbers'][idx]

            # Translate
            translate_inputs = trans_tokenizer(coco_eng_caption, return_tensors="pt")
            translated = trans_model.generate(**translate_inputs.to(args.device),
                                            forced_bos_token_id=trans_tokenizer.lang_code_to_id["lvs_Latn"])
            lv_caption = trans_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

            if caption_number == 1:
                all_caption = [] # reset all_caption
            all_caption.append(lv_caption)

            # Tokenize translated caption
            tokenized_caption = lv_tokenizer(lv_caption, padding='max_length', truncation=True,
                                            max_length=args.max_seq_len, return_tensors='pt')

            # Append the data to save_data
            save_data[split]['image_names'].append(image_name)
            save_data[split]['caption_numbers'].append(caption_number)
            save_data[split]['captions'].append(lv_caption)
            save_data[split]['all_captions'].append(all_caption)
            save_data[split]['input_ids'].append(tokenized_caption['input_ids'].squeeze())

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_TRANSLATED_LV.pkl'), 'wb') as f:
            pickle.dump(save_data[split], f)

    for split in ['valid', 'test']:
        # Save data as pickle file
        all_caption = []
        preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
        check_path(preprocessed_path)
        for idx in tqdm(range(len(loaded_data[split]['image_names'])), desc='Annotating Valid/Test Data with Translator...'):
            # Get image_name, caption
            image_name = loaded_data[split]['image_names'][idx]
            coco_eng_caption = loaded_data[split]['captions'][idx]
            caption_number = loaded_data[split]['caption_numbers'][idx]

            # Translate English to Latvian
            translate_inputs = trans2_tokenizer(coco_eng_caption, return_tensors="pt")
            translated = trans2_model.generate(**translate_inputs.to(args.device), forced_bos_token_id=trans2_tokenizer.lang_code_to_id["lv_LV"])
            lv_caption = trans2_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

            if caption_number == 1:
                all_caption = [] # reset all_caption
            all_caption.append(lv_caption)

            # Tokenize translated caption
            tokenized_caption = lv_tokenizer(lv_caption, padding='max_length', truncation=True,
                                            max_length=args.max_seq_len, return_tensors='pt')

            # Append the data to save_data
            save_data[split]['image_names'].append(image_name)
            save_data[split]['caption_numbers'].append(caption_number)
            save_data[split]['captions'].append(lv_caption)
            save_data[split]['all_captions'].append(all_caption)
            save_data[split]['input_ids'].append(tokenized_caption['input_ids'].squeeze())

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_TRANSLATED2_LV.pkl'), 'wb') as f:
            pickle.dump(save_data[split], f)

def translation_annotating_et(args: argparse.Namespace) -> None:
    # Define tokenizer
    trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(args.device)
    et_tokenizer = AutoTokenizer.from_pretrained('tartuNLP/EstBERT')

    # Define data_dict
    loaded_data = {}
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
        loaded_data['train'] = pickle.load(f)
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'valid_COCO_EN.pkl'), 'rb') as f:
        loaded_data['valid'] = pickle.load(f)
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'test_COCO_EN.pkl'), 'rb') as f:
        loaded_data['test'] = pickle.load(f)


    save_data = {
        'train': {
            'image_names': [],
            'caption_numbers': [],
            'captions': [],
            'all_captions': [],
            'input_ids': [],
            'tokenizer': et_tokenizer,
        },
        'valid': {
            'image_names': [],
            'caption_numbers': [],
            'captions': [],
            'all_captions': [],
            'input_ids': [],
            'tokenizer': et_tokenizer,
        },
        'test': {
            'image_names': [],
            'caption_numbers': [],
            'captions': [],
            'all_captions': [],
            'input_ids': [],
            'tokenizer': et_tokenizer,
        }
    }

    for split in ['train', 'valid', 'test']:
        # Save data as pickle file
        all_caption = []
        preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
        check_path(preprocessed_path)
        for idx in tqdm(range(len(loaded_data[split]['image_names'])), desc='Annotating with Translator...'):
            # Get image_name, caption
            image_name = loaded_data[split]['image_names'][idx]
            coco_eng_caption = loaded_data[split]['captions'][idx]
            caption_number = loaded_data[split]['caption_numbers'][idx]

            # Translate
            translate_inputs = trans_tokenizer(coco_eng_caption, return_tensors="pt")
            translated = trans_model.generate(**translate_inputs.to(args.device),
                                            forced_bos_token_id=trans_tokenizer.lang_code_to_id["est_Latn"])
            et_caption = trans_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

            if caption_number == 1:
                all_caption = [] # reset all_caption
            all_caption.append(et_caption)

            # Tokenize translated caption
            tokenized_caption = et_tokenizer(et_caption, padding='max_length', truncation=True,
                                            max_length=args.max_seq_len, return_tensors='pt')

            # Append the data to save_data
            save_data[split]['image_names'].append(image_name)
            save_data[split]['caption_numbers'].append(caption_number)
            save_data[split]['captions'].append(et_caption)
            save_data[split]['all_captions'].append(all_caption)
            save_data[split]['input_ids'].append(tokenized_caption['input_ids'].squeeze())

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_TRANSLATED_ET.pkl'), 'wb') as f:
            pickle.dump(save_data[split], f)
