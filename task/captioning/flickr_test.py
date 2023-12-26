# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# 3rd-party Modules
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from nlgeval import NLGEval
from bert_score import BERTScorer
from BARTScore.bart_score import BARTScorer
# Pytorch Modules
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    dataset_test = CaptioningDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_EN.pkl'), 'test')
    bart_src_lang, bart_tgt_lang = 'en_XX', 'en_XX'
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    tokenizer = dataset_test.tokenizer
    args.vocab_size = tokenizer.vocab_size
    args.bos_token_id = tokenizer.bos_token_id
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id
    image_transform = dataset_test.transform

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

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
        from wandb import AlertLevel
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
    test_acc_seq = 0
    result_df = pd.DataFrame(columns=['caption', 'reference', 'generated',
                                      'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
                                      'bleu_avg', 'rouge_l', 'meteor'])
    ref_list = []
    hyp_list = [] # For nlg-eval

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get input data from batch
        image_path = data_dicts['image_path']
        input_ids = data_dicts['input_ids'].to(device) # [batch_size, max_seq_len]
        target_ids = input_ids[:, 1:] # [batch_size, max_seq_len - 1] # Remove <bos> token
        caption = data_dicts['caption'] # list of str: for saving result
        all_caption = data_dicts['all_captions'] # list of list of str: for nlg-eval

        # Load image from image_path and make batch
        PIL_images = [Image.open(image_path).convert('RGB') for image_path in image_path]
        image = [image_transform(image) for image in PIL_images]
        image = torch.stack(image, dim=0).to(device) # [batch_size, 3, 224, 224]

        # Test - Forward pass
        with torch.no_grad():
            seq_output = model.inference(image) # [test_batch_size, max_seq_len]

        # Test - Calculate accuracy
        non_pad_mask = target_ids.ne(args.pad_token_id)
        batch_acc_seq = seq_output.eq(target_ids).masked_select(non_pad_mask).sum().item() / non_pad_mask.sum().item()

        # Test - Calculate bleu score
        batch_pred_ids = seq_output.cpu().numpy() # [test_batch_size, max_seq_len]
        batch_pred_sentences = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=False) # list of str
        for each_pred_sentence, each_reference in zip(batch_pred_sentences, all_caption):
            # If '</s>' is in the string, remove it and everything after it
            if '</s>' in each_pred_sentence: # facebook/bart-base
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('</s>')]
            elif '[EOS]' in each_pred_sentence: # cosmoquester/bart-ko-base
                each_pred_sentence = each_pred_sentence[:each_pred_sentence.index('[EOS]')]

            # Convert ' .' to '.' in reference - We need this trust me
            each_reference = [each_ref.replace(' .', '.') for each_ref in each_reference]

            result_df = result_df.append({'reference': each_reference,
                                          'generated': each_pred_sentence}, ignore_index=True)

            ref_list.append(each_reference) # multiple reference
            hyp_list.append(each_pred_sentence)

        # Test - Logging
        test_acc_seq += batch_acc_seq

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Acc: {batch_acc_seq:.4f}")

    # Test - Check accuracy
    test_acc_seq /= len(dataloader_test)

    # Test - nlg-eval
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    Eval = NLGEval(metrics_to_omit=['CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    BERT_Eval = BERTScorer(device=args.device, model_type='bert-base-multilingual-cased')
    BART_Eval = BARTScorer(device=args.device, checkpoint='facebook/mbart-large-50', source_lang=bart_src_lang, target_lang=bart_tgt_lang)

    # I don't know why but we need this
    _strip = lambda x: x.strip()
    ref_list2 = [list(map(_strip, refs)) for refs in zip(*ref_list)]
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

    # Final - End of testing
    write_log(logger, f"TEST - Acc: {test_acc_seq:.4f}")
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
        writer.add_scalar('TEST/Acc', test_acc_seq, global_step=0)
        writer.add_scalar('TEST/Bleu_1', metrics_dict['Bleu_1'], global_step=0)
        writer.add_scalar('TEST/Bleu_2', metrics_dict['Bleu_2'], global_step=0)
        writer.add_scalar('TEST/Bleu_3', metrics_dict['Bleu_3'], global_step=0)
        writer.add_scalar('TEST/Bleu_4', metrics_dict['Bleu_4'], global_step=0)
        writer.add_scalar('TEST/Bleu_avg', (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4, global_step=0)
        writer.add_scalar('TEST/Rouge_L', metrics_dict['ROUGE_L'], global_step=0)
        writer.add_scalar('TEST/Meteor', metrics_dict['METEOR'], global_step=0)
        writer.add_scalar('TEST/BERTScore_Precision', bert_score_P, global_step=0)
        writer.add_scalar('TEST/BERTScore_Recall', bert_score_R, global_step=0)
        writer.add_scalar('TEST/BERTScore_F1', bert_score_F1, global_step=0)
        writer.add_scalar('TEST/BARTScore', bart_score_total, global_step=0)

        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Annotation': [args.annotation_mode],
            'Decoding': [args.decoding_strategy],
            'Dec_arg': [args.beam_size if args.decoding_strategy == 'beam' else args.topk if args.decoding_strategy == 'topk' else args.topp if args.decoding_strategy == 'topp' else 0],
            'Acc': [test_acc_seq],
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

        # Send wandb alert
        if args.decoding_strategy == 'greedy':
            alert_text = f"TEST - Greedy: BLEU_Avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4}"
        elif args.decoding_strategy == 'multinomial':
            alert_text = f"TEST - Multinomial: BLEU_Avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4}"
        elif args.decoding_strategy == 'topk':
            alert_text = f"TEST - TopK:{args.topk}: BLEU_Avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4}"
        elif args.decoding_strategy == 'topp':
            alert_text = f"TEST - TopP:{args.topp}: BLEU_Avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4}"
        elif args.decoding_strategy == 'beam':
            alert_text = f"TEST - Beam:{args.beam_size}: BLEU_Avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4}"

        wandb.alert(
                title='Test End',
                text=alert_text,
                level=AlertLevel.INFO,
                wait_duration=300
        )

        wandb.finish()

    # Save result_df to csv file
    save_path = os.path.join(args.result_path, args.task, args.task_dataset)
    check_path(save_path)
    result_df.to_csv(os.path.join(args.result_path, args.task, args.task_dataset, f'result_{args.annotation_mode}_{args.decoding_strategy}.csv'), index=False)

    return test_acc_seq, metrics_dict
