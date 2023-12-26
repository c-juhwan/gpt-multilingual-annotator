# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from nlgeval import NLGEval
from bert_score import BERTScorer
from BARTScore.bart_score import BARTScorer
# Pytorch Modules
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
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
    if args.annotation_mode in ['original_en', 'gpt_en', 'backtrans_en', 'eda_en', 'synonym_en', 'hrqvae_en', 'onlyone_en', 'budget_en']:
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
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args) + f' - Test: {args.decoding_strategy}',
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Annotation: {args.annotation_mode}",
                         f"Encoder: {args.encoder_type}",
                         f"Decoder: {args.decoder_type}"])

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
    if args.annotation_mode in ['original_en', 'gpt_en', 'backtrans_en', 'eda_en', 'synonym_en', 'hrqvae_en', 'onlyone_en', 'budget_en']:
        valid_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                      f'captions_val2014_{args.annotation_mode}_{args.decoding_strategy}_results.json'), orient='records')
        test_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                     f'captions_test2014_{args.annotation_mode}_{args.decoding_strategy}_results.json'), orient='records')
    elif args.annotation_mode in ['aihub_ko', 'gpt_ko']:
        valid_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                      f'captions_val2014_{args.annotation_mode}_{args.decoding_strategy}_results_ko.json'), orient='records', force_ascii=False)
        test_df.to_json(os.path.join(args.result_path, args.task, args.task_dataset, args.annotation_mode,
                                     f'captions_test2014_{args.annotation_mode}_{args.decoding_strategy}_results_ko.json'), orient='records', force_ascii=False)
        translate_to_eng(args, valid_df, test_df)
        evaluate_kor_valid(args, valid_df, dataset_dict['valid'], logger, writer)

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

def evaluate_kor_valid(args: argparse.Namespace, valid_df: pd.DataFrame, valid_dataset: Dataset, logger, writer) -> None:
    Eval = NLGEval(metrics_to_omit=['CIDEr', 'SPICE', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    BERT_Eval = BERTScorer(device=args.device, model_type='bert-base-multilingual-cased')
    BART_Eval = BARTScorer(device=args.device, checkpoint='facebook/mbart-large-50', source_lang='ko_KR', target_lang='ko_KR')

    valid_dataset = valid_dataset.data_list

    # Get valid_df['image_id'] and valid_df['caption']
    ref_list = []
    hyp_list = []

    for idx in range(len(valid_df)):
        image_id = valid_df.loc[idx, 'image_id']

        image_path = valid_dataset[idx]['image_path']
        image_id_ = os.path.basename(image_path).split('.')[0]
        if args.task_dataset == 'coco2014':
            image_id_ = image_id_.split('_')[-1] # Remove leading 'COCO_val2014_' from image_id
        image_id_ = int(image_id_)

        assert image_id == image_id_ # Check if image_id is correct

        caption = valid_df.loc[idx, 'caption']
        ref_list.append([valid_dataset[idx]['caption']])
        hyp_list.append(caption)

    # Convert ' .' in reference to '.' - I don't know why but we need to do this, otherwise it will give error
    replace_lambda = lambda x: x.replace(' .', '.')
    ref_list2 = [list(map(replace_lambda, refs)) for refs in zip(*ref_list)]

    metrics_dict = Eval.compute_metrics(ref_list2, hyp_list)
    bert_score_P, bert_score_R, bert_score_F1, bart_score_total = 0, 0, 0, 0

    for each_ref, each_hyp in tqdm(zip(ref_list2[0], hyp_list), total=len(ref_list2[0]), desc=f'TEST - Calculating BERTScore&BARTScore...'):
        P, R, F1 = BERT_Eval.score([each_ref], [each_hyp])
        bert_score_P += P.item()
        bert_score_R += R.item()
        bert_score_F1 += F1.item()
        bart_score = BART_Eval.multi_ref_score([each_ref], [each_hyp], agg='max')
        bart_score_total += bart_score[0].item()
    bert_score_P /= len(ref_list2[0])
    bert_score_R /= len(ref_list2[0])
    bert_score_F1 /= len(ref_list2[0])
    bart_score_total /= len(ref_list2[0])

    write_log(logger, f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
    write_log(logger, f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
    write_log(logger, f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
    write_log(logger, f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
    write_log(logger, f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
    write_log(logger, f"TEST - Rouge_L: {metrics_dict['ROUGE_L']:.4f}")
    write_log(logger, f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")
    write_log(logger, f"TEST - BERTScore_Precision: {bert_score_P:.4f}")
    write_log(logger, f"TEST - BERTScore_Recall: {bert_score_R:.4f}")
    write_log(logger, f"TEST - BERTScore_F1: {bert_score_F1:.4f}")
    write_log(logger, f"TEST - BARTScore: {bart_score_total:.4f}")

    if args.use_tensorboard:
        writer.add_scalar('TEST/KoVal_Bleu_1', metrics_dict['Bleu_1'], global_step=0)
        writer.add_scalar('TEST/KoVal_Bleu_2', metrics_dict['Bleu_2'], global_step=0)
        writer.add_scalar('TEST/KoVal_Bleu_3', metrics_dict['Bleu_3'], global_step=0)
        writer.add_scalar('TEST/KoVal_Bleu_4', metrics_dict['Bleu_4'], global_step=0)
        writer.add_scalar('TEST/KoVal_Bleu_avg', (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4, global_step=0)
        writer.add_scalar('TEST/KoVal_Rouge_L', metrics_dict['ROUGE_L'], global_step=0)
        writer.add_scalar('TEST/KoVal_Meteor', metrics_dict['METEOR'], global_step=0)
        writer.add_scalar('TEST/KoVal_BERTScore_Precision', bert_score_P, global_step=0)
        writer.add_scalar('TEST/KoVal_BERTScore_Recall', bert_score_R, global_step=0)
        writer.add_scalar('TEST/KoVal_BERTScore_F1', bert_score_F1, global_step=0)
        writer.add_scalar('TEST/KoVal_BARTScore', bart_score_total, global_step=0)

    if args.use_wandb:
        import wandb
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Annotation': [args.annotation_mode],
            'Decoding': [args.decoding_strategy],
            'Dec_arg': [args.beam_size if args.decoding_strategy == 'beam' else args.topk if args.decoding_strategy == 'topk' else args.topp if args.decoding_strategy == 'topp' else 0],
            'Bleu_1': [metrics_dict['Bleu_1']],
            'Bleu_2': [metrics_dict['Bleu_2']],
            'Bleu_3': [metrics_dict['Bleu_3']],
            'Bleu_4': [metrics_dict['Bleu_4']],
            'Bleu_avg': [(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4],
            'Rouge_L': [metrics_dict['ROUGE_L']],
            'Meteor': [metrics_dict['METEOR']],
            'BERTScore_Precision': [bert_score_P],
            'BERTScore_Recall': [bert_score_R],
            'BERTScore_F1': [bert_score_F1],
            'BARTScore': [bart_score_total],
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
