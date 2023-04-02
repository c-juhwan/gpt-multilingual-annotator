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
from nlgeval import NLGEval
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
    dataset_test = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_processed.pkl'), 'test')
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    args.vocab_size = tokenizer.vocab_size
    args.bos_token_id = tokenizer.bos_token_id
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

    # Get model instance
    write_log(logger, "Building model")
    model = CaptioningModel(args).to(device)

    # Define loss function
    seq_loss = nn.CrossEntropyLoss(ignore_index=args.pad_token_id,
                                   label_smoothing=args.label_smoothing_eps)

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'{args.encoder_type}_{args.decoder_type}_final_model.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Test - Start evaluation
    model = model.eval()
    test_acc_seq = 0
    result_df = pd.DataFrame(columns=['caption', 'reference', 'generated',
                                      'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
                                      'bleu_avg', 'rouge_l', 'meteor'])
    ref_list = []
    hyp_list = [] # For nlg-eval

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get input data from batch
        image = data_dicts['image'].to(device) # [test_batch_size, 3, 224, 224]
        caption = data_dicts['caption'] # list of str: for saving result
        #all_caption = data_dicts['all_caption'] # list of list of str: for calculating bleu score, used as reference
        input_ids = data_dicts['input_ids'].to(device) # [test_batch_size, max_seq_len]
        target_ids = input_ids[:, 1:] # [test_batch_size, max_seq_len - 1] # Remove <bos> token

        # Test - Forward pass
        with torch.no_grad():
            seq_output = model.inference(image) # [test_batch_size, max_seq_len]

        # Test - Calculate accuracy
        non_pad_mask = target_ids.ne(args.pad_token_id)
        batch_acc_seq = seq_output.eq(target_ids).masked_select(non_pad_mask).sum().item() / non_pad_mask.sum().item()

        # Test - Calculate bleu score
        batch_pred_ids = seq_output.cpu().numpy() # [test_batch_size, max_seq_len]
        batch_pred_sentences = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=False) # list of str
        for each_pred_sentence, each_reference in zip(batch_pred_sentences, caption):
            # If '</s>' is in the string, remove it and everything after it
            if '</s>' in each_pred_sentence:
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('</s>')]

            # Convert ' .' to '.' in reference - We need this trust me
            each_reference = each_reference.replace(' .', '.')

            result_df = result_df.append({'reference': each_reference,
                                          'generated': each_pred_sentence}, ignore_index=True)

            ref_list.append([each_reference])
            hyp_list.append(each_pred_sentence)

        #raise Exception # For debugging

        # Test - Logging
        test_acc_seq += batch_acc_seq

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Acc: {batch_acc_seq:.4f}")

    # Test - Check accuracy
    test_acc_seq /= len(dataloader_test)

    # Test - nlg-eval
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    Eval = NLGEval(metrics_to_omit=['CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])

    # Convert ' .' in reference to '.' - I don't know why but we need to do this, otherwise it will give error
    replace_lambda = lambda x: x.replace(' .', '.')
    ref_list2 = [list(map(replace_lambda, refs)) for refs in zip(*ref_list)]

    metrics_dict = Eval.compute_metrics(ref_list2, hyp_list)
    print(metrics_dict)

    # Final - End of testing
    write_log(logger, f"TEST - Acc: {test_acc_seq:.4f}")
    write_log(logger, f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
    write_log(logger, f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
    write_log(logger, f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
    write_log(logger, f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
    write_log(logger, f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
    write_log(logger, f"TEST - Rouge_L: {metrics_dict['ROUGE_L']:.4f}")
    write_log(logger, f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")

    if args.use_tensorboard:
        writer.add_scalar('TEST/Acc', test_acc_seq, global_step=0)
        writer.add_scalar('TEST/Bleu_1', metrics_dict['Bleu_1'], global_step=0)
        writer.add_scalar('TEST/Bleu_2', metrics_dict['Bleu_2'], global_step=0)
        writer.add_scalar('TEST/Bleu_3', metrics_dict['Bleu_3'], global_step=0)
        writer.add_scalar('TEST/Bleu_4', metrics_dict['Bleu_4'], global_step=0)
        writer.add_scalar('TEST/Bleu_avg', (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4, global_step=0)
        writer.add_scalar('TEST/Rouge_L', metrics_dict['ROUGE_L'], global_step=0)
        writer.add_scalar('TEST/Meteor', metrics_dict['METEOR'], global_step=0)

        writer.close()

    # Save result_df to csv file
    save_path = os.path.join(args.result_path, args.task, args.task_dataset)
    check_path(save_path)
    result_df.to_csv(os.path.join(args.result_path, args.task, args.task_dataset, 'result.csv'), index=False)

    return test_acc_seq, metrics_dict
