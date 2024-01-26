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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path
from task.captioning.preprocessing import load_caption_data

def translation_annotating_vie(args: argparse.Namespace) -> None:
    # Define tokenizer
    trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(args.device)
    vie_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    # Define data_dict
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_COCO_EN.pkl'), 'rb') as f:
        loaded_data = pickle.load(f)

    save_data = {
        'image_names': [],
        'caption_numbers': [],
        'captions': [],
        'all_captions': [],
        'input_ids': [],
        'tokenizer': vie_tokenizer,
    }

    # Save data as pickle file
    all_caption = []
    preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
    check_path(preprocessed_path)
    for idx in tqdm(range(len(loaded_data['image_names'])), desc='Annotating with Translator...'):
        # Get image_name, caption
        image_name = loaded_data['image_names'][idx]
        coco_eng_caption = loaded_data['captions'][idx]
        caption_number = loaded_data['caption_numbers'][idx]

        # Translate
        translate_inputs = trans_tokenizer(coco_eng_caption, return_tensors="pt")
        translated = trans_model.generate(**translate_inputs.to(args.device),
                                          forced_bos_token_id=trans_tokenizer.lang_code_to_id["vie_Latn"])
        vie_caption = trans_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

        if caption_number == 1:
            all_caption = [] # reset all_caption
        all_caption.append(vie_caption)

        # Tokenize translated caption
        tokenized_caption = vie_tokenizer(vie_caption, padding='max_length', truncation=True,
                                          max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to save_data
        save_data['image_names'].append(image_name)
        save_data['caption_numbers'].append(caption_number)
        save_data['captions'].append(vie_caption)
        save_data['all_captions'].append(all_caption)
        save_data['input_ids'].append(tokenized_caption['input_ids'].squeeze())

    # Save data as pickle file
    with open(os.path.join(preprocessed_path, 'train_TRANSLATED_VIE.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
