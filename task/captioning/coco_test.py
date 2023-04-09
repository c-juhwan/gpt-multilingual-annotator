# Standard Library Modules
import os
import sys
import random
import logging
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from multiprocessing import Pool
# 3rd-party Modules
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from nlgeval import NLGEval
from googletrans import Translator
# Pytorch Modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.captioning.model import CaptioningModel
from model.captioning.dataset import CaptioningDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_torch_device, check_path

def testing(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Initialize tensorboard writer
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset...")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['test'] = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_EN.pkl'), 'test')
    if args.annotation_mode in ['original_en', 'gpt_en', 'backtrans_en']:
        dataset_dict['valid']  = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_EN.pkl'), 'valid')
    elif args.annotation_mode in ['aihub_ko', 'gpt_ko', 'backtrans_ko']:
        dataset_dict['valid']  = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_AIHUB_KO.pkl'), 'valid')

    dataloader_dict['test'] = DataLoader(dataset_dict['test'], batch_size=args.test_batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.test_batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    tokenizer = dataset_dict['valid']['tokenizer'] # Depends on annotation_mode -> language
    args.vocab_size = tokenizer.vocab_size
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id
    image_transform = dataset_dict['valid'].transform

    write_log(logger, "Loaded data successfully")

    # Get model instance
    write_log(logger, "Building model")
    model = CaptioningModel(args).to(device)

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'{args.encoder_type}_{args.decoder_type}_final_model_{args.annotation_mode}.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Test - Start evaluation
    model = model.eval()
    valid_df = pd.DataFrame(columns=['image_id', 'caption'])
    test_df = pd.DataFrame(columns=['image_id', 'caption'])

    for valid_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc=f'Testing on COCO valid')):
        # Test - Get input data from batch
        image_path = data_dicts['image_path']
        # Get image id from image path
        image_id = [os.path.basename(image_path).split('.')[0] for image_path in image_path]
        image_id = [int(image_id) for image_id in image_id] # Remove leading zeros

        # Load image from image_path and make batch
        PIL_images = [Image.open(image_path).convert('RGB') for image_path in image_path]
        image = [image_transform(image) for image in PIL_images]
        image = torch.stack(image, dim=0).to(device) # [batch_size, 3, 224, 224]

        # Test - Forward pass
        with torch.no_grad():
            seq_output = model.inference(image) # [test_batch_size, max_seq_len]

        # Test - Calculate bleu score
        batch_pred_ids = seq_output.cpu().numpy() # [test_batch_size, max_seq_len]
        batch_pred_sentences = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=False) # list of str
        for each_pred_sentence, each_id in zip(batch_pred_sentences, image_id):
            # If '</s>' is in the string, remove it and everything after it
            if '</s>' in each_pred_sentence:
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('</s>')]

            valid_df = valid_df.append({'image_id': each_id,
                                        'caption': each_pred_sentence}, ignore_index=True)

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['test'], total=len(dataloader_dict['test']), desc=f'Testing on COCO test')):
        # Test - Get input data from batch
        image_path = data_dicts['image_path']
        # Get image id from image path
        image_id = [os.path.basename(image_path).split('.')[0] for image_path in image_path]
        image_id = [int(image_id) for image_id in image_id] # Remove leading zeros

        # Load image from image_path and make batch
        PIL_images = [Image.open(image_path).convert('RGB') for image_path in image_path]
        image = [image_transform(image) for image in PIL_images]
        image = torch.stack(image, dim=0).to(device) # [batch_size, 3, 224, 224]

        # Test - Forward pass
        with torch.no_grad():
            seq_output = model.inference(image) # [test_batch_size, max_seq_len]

        # Test - Calculate bleu score
        batch_pred_ids = seq_output.cpu().numpy() # [test_batch_size, max_seq_len]
        batch_pred_sentences = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=False) # list of str
        for each_pred_sentence, each_id in zip(batch_pred_sentences, image_id):
            # If '</s>' is in the string, remove it and everything after it
            if '</s>' in each_pred_sentence:
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('</s>')]

            test_df = test_df.append({'image_id': each_id,
                                      'caption': each_pred_sentence}, ignore_index=True)


    # Save valid_df and test_df to json file for coco evaluation
    check_path(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode))
    if args.annotation_mode in ['original_en', 'gpt_en', 'backtrans_en']:
        valid_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                      f'captions_val2014_{args.annotation_mode}_results.json'), orient='records')
        test_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                     f'captions_test2014_{args.annotation_mode}_results.json'), orient='records')
    elif args.annotation_mode in ['aihub_ko', 'gpt_ko', 'backtrans_ko']:
        valid_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                      f'captions_val2014_{args.annotation_mode}_results_ko.json'), orient='records')
        test_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                     f'captions_test2014_{args.annotation_mode}_results_ko.json'), orient='records')
        translate_to_eng(args, valid_df, test_df)

NUM_PROCESS = 4
tqdm_bar = tqdm(total=100, desc='Progress', position=0)
def translate_to_eng(args: argparse.Namespace, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:


    # Define Translators
    random_port = [
        (random.randint(1000, 9999), random.randint(1000, 9999)) for _ in range(NUM_PROCESS)
    ]
    translators = [
        Translator(url=['translate.google.com', 'translate.google.co.kr'],
                   proxies={'http': f'127.0.0.1:{port[0]}', 'http://host.name': f'127.0.0.1:{port[1]}'}) for port in random_port
    ]

    # Split valid_df and test_df into NUM_PROCESS parts
    valid_df_subset = np.array_split(valid_df, NUM_PROCESS)
    test_df_subset = np.array_split(test_df, NUM_PROCESS)

    # Reset index of valid_df_subset and test_df_subset
    for i in range(NUM_PROCESS):
        valid_df_subset[i].reset_index(drop=True, inplace=True)
        test_df_subset[i].reset_index(drop=True, inplace=True)

    # Call multiprocessing
    test_starmap_items = [
        (test_df_subset[i], translators[i]) for i in range(NUM_PROCESS)
    ]
    valid_starmap_items = [
        (valid_df_subset[i], translators[i]) for i in range(NUM_PROCESS)
    ]

    print(f"Start multiprocessing with {NUM_PROCESS} processes")

    with Pool(NUM_PROCESS) as p:
        test_result = p.starmap(try_call_trans, test_starmap_items)
    with Pool(NUM_PROCESS) as p:
        valid_result = p.starmap(try_call_trans, valid_starmap_items)

    # Concatenate valid_result and test_result
    valid_df_translated = []
    test_df_translated = []
    for i in range(NUM_PROCESS):
        valid_df_translated += valid_result[i]
        test_df_translated += test_result[i]

    # Convert to DataFrame
    valid_df_translated = pd.DataFrame(valid_df_translated)
    test_df_translated = pd.DataFrame(test_df_translated)

    # Save valid_df_translated and test_df_translated to json file for coco evaluation
    check_path(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode))
    valid_df_translated.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                             f'captions_val2014_{args.annotation_mode}_results.json'), orient='records')
    test_df_translated.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                            f'captions_test2014_{args.annotation_mode}_results.json'), orient='records')

    tqdm_bar.close()


def try_call_trans(df_subset: pd.DataFrame, translator) -> list:
    try:
        return call_trans(df_subset, translator)
    except KeyboardInterrupt as k:
        raise k
    except Exception as e:
        logging.exception(f"Error in try_call_trans: {e}")

def call_trans(df_subset: pd.DataFrame, translator) -> list:
    subset_list = []

    for idx in tqdm(range(len(df_subset)), desc='Translating to English'):
        # Get image id and caption
        image_id = df_subset.loc[idx, 'image_id']
        caption = df_subset.loc[idx, 'caption']

        # Translate to English
        translated_caption = translator.translate(caption, dest='en').text

        # Append to subset_list
        subset_list.append({'image_id': image_id, 'caption': translated_caption})

    tqdm_bar.update(1)