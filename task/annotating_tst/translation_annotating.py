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
from task.text_style_transfer.preprocessing import get_dataset_path

def translation_annotating(args: argparse.Namespace) -> None:
    """
    Load GYAFC (English) data and translate it to target language.
    """

    # Define tokenizer
    trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(args.device)
    _, lang_code = get_dataset_path(args)
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=lang_code, tgt_lang=lang_code)

    if lang_code == 'pt_XX':
        trans_lang_code = trans_tokenizer.lang_code_to_id["por_Latn"]
        out_lang_code = 'PT'
    elif lang_code == 'fr_XX':
        trans_lang_code = trans_tokenizer.lang_code_to_id["fra_Latn"]
        out_lang_code = 'FR'
    elif lang_code == 'it_IT':
        trans_lang_code = trans_tokenizer.lang_code_to_id["ita_Latn"]
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
    for idx in tqdm(range(len(gyafc_data['informal_text'])), desc='Annotating with Translator...'):
        # Get data
        informal_text = gyafc_data['informal_text'][idx]
        formal_text = gyafc_data['formal_text'][idx]
        text_number = gyafc_data['text_number'][idx]
        category = gyafc_data['category'][idx]

        # Translate
        translate_inputs_informal = trans_tokenizer(informal_text, return_tensors="pt")
        translate_inputs_formal = trans_tokenizer(formal_text, return_tensors="pt")

        translated_informal = trans_model.generate(**translate_inputs_informal.to(args.device),
                                                   forced_bos_token_id=trans_lang_code)
        target_informal = trans_tokenizer.batch_decode(translated_informal, skip_special_tokens=True)[0]

        translated_formal = trans_model.generate(**translate_inputs_formal.to(args.device),
                                                 forced_bos_token_id=trans_lang_code)
        target_formal = trans_tokenizer.batch_decode(translated_formal, skip_special_tokens=True)[0]

        if text_number == 1:
            all_references = []
        all_references.append(target_formal)

        # Tokenize translated text
        tokenized = tokenizer(target_informal, text_target=target_formal, padding='max_length', truncation=True,
                              max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to the data_dict
        save_data['informal_text'].append(target_informal)
        save_data['formal_text'].append(target_formal)
        save_data['all_references'].append(all_references)
        save_data['text_number'].append(text_number)
        save_data['category'].append(category)
        save_data['model_input_ids'].append(tokenized['input_ids'].squeeze())

    # Save data as pickle file
    with open(os.path.join(preprocessed_path, f'train_TRANSLATED_{out_lang_code}.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
        print(f"Saved train_TRANSLATED_{out_lang_code}.pkl in {preprocessed_path}")
