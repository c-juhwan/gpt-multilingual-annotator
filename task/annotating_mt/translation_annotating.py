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

def translation_annotating(args: argparse.Namespace) -> None:
    """
    Load GYAFC (English) data and translate it to target language.
    """

    # Define tokenizer
    trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(args.device)

    # Define language code
    if args.annotation_mode == 'translated_de':
        lang_code = 'de_DE'
        trans_lang_code = trans_tokenizer.lang_code_to_id["deu_Latn"]
        out_lang_code = 'DE'
    if args.annotation_mode == 'translated_lv': # Latvian
        lang_code = 'lv_LV'
        trans_lang_code = trans_tokenizer.lang_code_to_id["lvs_Latn"]
        out_lang_code = 'LV'
        trans2_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="lv_LV")
        trans2_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(args.device)
    if args.annotation_mode == 'translated_et': # Estonian
        lang_code = 'et_EE'
        trans_lang_code = trans_tokenizer.lang_code_to_id["est_Latn"]
        out_lang_code = 'ET'
        trans2_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="et_EE")
        trans2_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(args.device)
    if args.annotation_mode == 'translated_fi': # Finnish
        lang_code = 'fi_FI'
        trans_lang_code = trans_tokenizer.lang_code_to_id["fin_Latn"]
        out_lang_code = 'FI'
        trans2_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fi_FI")
        trans2_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(args.device)

    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang=lang_code, tgt_lang=lang_code)

    # Define data_dict
    loaded_data = {}
    with open(os.path.join(args.preprocess_path, 'machine_translation', args.task_dataset, 'train_ORIGINAL_DE.pkl'), 'rb') as f:
        loaded_data['train'] = pickle.load(f)
    with open(os.path.join(args.preprocess_path, 'machine_translation', args.task_dataset, 'valid_ORIGINAL_DE.pkl'), 'rb') as f:
        loaded_data['valid'] = pickle.load(f)
    with open(os.path.join(args.preprocess_path, 'machine_translation', args.task_dataset, 'test_ORIGINAL_DE.pkl'), 'rb') as f:
        loaded_data['test'] = pickle.load(f)

    save_data = {
        'train': {
            'source_text': [],
            'target_text': [],
            'text_number': [],
            'model_input_ids': [],
            'tokenizer': tokenizer,
        },
        'valid': {
            'source_text': [],
            'target_text': [],
            'text_number': [],
            'model_input_ids': [],
            'tokenizer': tokenizer,
        },
        'test': {
            'source_text': [],
            'target_text': [],
            'text_number': [],
            'model_input_ids': [],
            'tokenizer': tokenizer,
        },
    }

    # Save data as pickle file
    all_references = []
    preprocessed_path = os.path.join(args.preprocess_path, 'machine_translation', args.task_dataset)
    check_path(preprocessed_path)
    for split in ['train']: # Train data will be annotated with NLLB
        for idx in tqdm(range(len(loaded_data[split]['source_text'])), desc=f'Annotating {split} data with Translator...'):
            # Get data
            source_text = loaded_data[split]['source_text'][idx]
            # target_text = loaded_data[split]['target_text'][idx]
            text_number = loaded_data[split]['text_number'][idx]

            # Translate
            translate_inputs_source = trans_tokenizer(source_text, return_tensors="pt")

            translated_target = trans_model.generate(**translate_inputs_source.to(args.device),
                                                    forced_bos_token_id=trans_lang_code)
            translated_target_text = trans_tokenizer.batch_decode(translated_target, skip_special_tokens=True)[0]

            # Tokenize translated text
            tokenized = tokenizer(source_text, text_target=translated_target_text, padding='max_length', truncation=True,
                                max_length=args.max_seq_len, return_tensors='pt')

            # Append the data to the data_dict
            save_data[split]['source_text'].append(source_text)
            save_data[split]['target_text'].append(translated_target_text)
            save_data[split]['text_number'].append(text_number)
            save_data[split]['model_input_ids'].append(tokenized['input_ids'].squeeze())

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_TRANSLATED_{out_lang_code}.pkl'), 'wb') as f:
            pickle.dump(save_data[split], f)

    for split in ['valid', 'test']: # Valid and test data will be annotated with MBart
        for idx in tqdm(range(len(loaded_data[split]['source_text'])), desc=f'Annotating {split} data with Translator...'):
            # Get data
            source_text = loaded_data[split]['source_text'][idx]
            # target_text = loaded_data[split]['target_text'][idx]
            text_number = loaded_data[split]['text_number'][idx]

            # Translate English to Target Language
            translate_inputs = trans2_tokenizer(source_text, return_tensors="pt")
            translated = trans2_model.generate(**translate_inputs.to(args.device),
                                                forced_bos_token_id=trans2_tokenizer.lang_code_to_id[f"{lang_code}"])
            translated_target_text = trans2_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

            # Tokenize translated caption
            tokenized = tokenizer(source_text, text_target=translated_target_text, padding='max_length', truncation=True,
                                max_length=args.max_seq_len, return_tensors='pt')

            # Append the data to the data_dict
            save_data[split]['source_text'].append(source_text)
            save_data[split]['target_text'].append(translated_target_text)
            save_data[split]['text_number'].append(text_number)
            save_data[split]['model_input_ids'].append(tokenized['input_ids'].squeeze())

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_TRANSLATED_{out_lang_code}.pkl'), 'wb') as f:
            pickle.dump(save_data[split], f)
