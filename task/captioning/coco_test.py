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
from easynmt import EasyNMT
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
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

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
                                          shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.test_batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    tokenizer = dataset_dict['valid'].tokenizer # Depends on annotation_mode -> language
    args.vocab_size = tokenizer.vocab_size
    args.bos_token_id = tokenizer.bos_token_id
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id
    image_transform = dataset_dict['valid'].transform

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Number of valid samples: {len(dataset_dict['valid'])} | Number of test samples: {len(dataset_dict['test'])}")

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

    # Load Wandb
    if args.use_wandb:
        import wandb
        wandb.init(
                project=args.proj_name,
                name=get_wandb_exp_name(args),
                config=args,
                tags=[f"Dataset: {args.task_dataset}",
                      f"Annotation: {args.annotation_mode}",
                      f"Encoder: {args.encoder_type}",
                      f"Decoder: {args.decoder_type}",
                      f"Desc: {args.description}"],
                resume=True,
                id=checkpoint['wandb_id']
            )

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
        if args.task_dataset == 'coco2014':
            image_id = [image_id.split('_')[-1] for image_id in image_id] # Remove leading 'COCO_val2014_' from image_id
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
            if '</s>' in each_pred_sentence: # facebook/bart-base
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('</s>')]
            elif '[EOS]' in each_pred_sentence: # cosmoquester/bart-ko-base
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('[EOS]')]

            valid_df = valid_df.append({'image_id': each_id,
                                        'caption': each_pred_sentence}, ignore_index=True)

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['test'], total=len(dataloader_dict['test']), desc=f'Testing on COCO test')):
        # Test - Get input data from batch
        image_path = data_dicts['image_path']
        # Get image id from image path
        image_id = [os.path.basename(image_path).split('.')[0] for image_path in image_path]
        if args.task_dataset == 'coco2014':
            image_id = [image_id.split('_')[-1] for image_id in image_id] # Remove leading 'COCO_test2014_' from image_id
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
            elif '[EOS]' in each_pred_sentence: # cosmoquester/bart-ko-base
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('[EOS]')]

            test_df = test_df.append({'image_id': each_id,
                                      'caption': each_pred_sentence}, ignore_index=True)

    # Save valid_df and test_df to json file for coco evaluation
    check_path(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode))
    if args.annotation_mode in ['original_en', 'gpt_en', 'backtrans_en']:
        valid_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                      f'captions_val2014_{args.annotation_mode}_{args.decoding_strategy}_results.json'), orient='records')
        test_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                     f'captions_test2014_{args.annotation_mode}_{args.decoding_strategy}_results.json'), orient='records')
    elif args.annotation_mode in ['aihub_ko', 'gpt_ko', 'backtrans_ko']:
        valid_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                      f'captions_val2014_{args.annotation_mode}_{args.decoding_strategy}_results_ko.json'), orient='records', force_ascii=False)
        test_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                     f'captions_test2014_{args.annotation_mode}_{args.decoding_strategy}_results_ko.json'), orient='records', force_ascii=False)
        translate_to_eng(args, valid_df, test_df)

    if args.use_wandb:
        wandb.finish() # Finish wandb run -> send alert

def translate_to_eng(args: argparse.Namespace, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    nmt_model = EasyNMT('mbart50_m2m')

    # Translate valid_df and test_df to English
    valid_df_translated = valid_df.copy()
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc='Translating valid_df to English'):
        valid_df_translated.loc[idx, 'caption'] = nmt_model.translate(row['caption'], target_lang='en')
    test_df_translated = test_df.copy()
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Translating test_df to English'):
        test_df_translated.loc[idx, 'caption'] = nmt_model.translate(row['caption'], target_lang='en')

    assert len(valid_df_translated) == len(valid_df)
    assert len(test_df_translated) == len(test_df)

    # Save valid_df_translated and test_df_translated to json file for coco evaluation
    check_path(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode))
    valid_df_translated.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                             f'captions_val2014_{args.annotation_mode}_{args.decoding_strategy}_results.json'), orient='records')
    test_df_translated.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                            f'captions_test2014_{args.annotation_mode}_{args.decoding_strategy}_results.json'), orient='records')
