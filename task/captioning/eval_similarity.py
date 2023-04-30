# Standard Library Modules
import os
import sys
import logging
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
# Pytorch Modules
import torch
import torch.nn as nn
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.captioning.model import CaptioningModel
from model.captioning.dataset import CaptioningDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

def eval_similarity(args):
    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load checkpoint
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'{args.encoder_type}_{args.decoder_type}_final_model_{args.annotation_mode}.pt')
    checkpoint = torch.load(load_model_name, map_location='cpu')

    # Load Wandb
    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
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

    model = SentenceTransformer('all-MiniLM-L6-v2')

    if args.annotation_mode == 'original_en':
        train_dataset = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_ORIGINAL_EN.pkl'), 'train')
    elif args.annotation_mode == 'gpt_en' and args.gpt_model_version == 'gpt-3.5-turbo':
        train_dataset = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_GPT35_EN.pkl'), 'train')
    elif args.annotation_mode == 'gpt_en' and args.gpt_model_version == 'gpt-4':
        train_dataset = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_GPT4_EN.pkl'), 'train')
    elif args.annotation_mode == 'backtrans_en':
        train_dataset = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_BT_EN.pkl'), 'train')
    elif args.annotation_mode == 'eda_en':
        train_dataset = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_EDA_EN.pkl'), 'train')
    elif args.annotation_mode == 'synonym_en':
        train_dataset = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_SR_EN.pkl'), 'train')
    elif args.annotation_mode == 'hrqvae_en':
        train_dataset = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_HRQ_EN.pkl'), 'train')
    else:
        raise ValueError(f"Invalid annotation mode for this job: {args.annotation_mode}")

    train_data_list = train_dataset.data_list

    # Get gold (caption_number == 1)
    idx = 0
    total_similarity = 0
    sample_count = 0
    tqdm_bar = tqdm(total=len(train_data_list), desc='Calculating similarity', position=0, leave=True)

    # sort data_list by image_path
    train_data_list = sorted(train_data_list, key=lambda x: x['image_path'])

    while idx < len(train_data_list):
        if train_data_list[idx]['caption_number'] == 1:
            gold = train_data_list[idx]
            preds = train_data_list[idx+1:idx+5]

            assert gold['image_path'] == preds[0]['image_path'] == preds[1]['image_path'] == preds[2]['image_path'] == preds[3]['image_path']

            gold_embedding = model.encode(gold['caption'], convert_to_tensor=True).to('cpu')
            similarity = 0

            for i in range(4):
                pred_embedding = model.encode(preds[i]['caption'], convert_to_tensor=True).to('cpu')
                similarity += util.cos_sim(gold_embedding, pred_embedding)

            total_similarity += similarity / 4
            idx += 5 # Skip 4 other captions
            sample_count += 1
            tqdm_bar.update(5)
        else:
            idx += 1
            continue

    average_similarity = total_similarity / sample_count
    write_log(logger, f"Average similarity: {average_similarity}")

    if args.use_wandb:
        wandb.log({'Average similarity': average_similarity})
        wandb.finish()
