# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import logging
import argparse
# 3rd-party Modules
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
# Huggingface Modules
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.machine_translation.dataset import MTDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

def testing(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)
    assert args.test_batch_size == 1, "Test batch size must be 1"

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
    if args.annotation_mode in ['original_de', 'translated_de', 'gpt_de', 'googletrans_de']:
        dataset_test = MTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_DE.pkl'), 'test')
        bart_src_lang, bart_tgt_lang = 'en_XX', 'de_DE'
    elif args.annotation_mode in ['translated_lv', 'gpt_lv', 'googletrans_lv']:
        dataset_test = MTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_TRANSLATED_LV.pkl'), 'test')
        bart_src_lang, bart_tgt_lang = 'en_XX', 'lv_LV'
    elif args.annotation_mode in ['translated_et', 'gpt_et', 'googletrans_et']:
        dataset_test = MTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_TRANSLATED_ET.pkl'), 'test')
        bart_src_lang, bart_tgt_lang = 'en_XX', 'et_EE'
    elif args.annotation_mode in ['translated_fi', 'gpt_fi', 'googletrans_fi']:
        dataset_test = MTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_TRANSLATED_FI.pkl'), 'test')
        bart_src_lang, bart_tgt_lang = 'en_XX', 'fi_FI'
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    tokenizer = dataset_test.tokenizer
    lang_id = tokenizer.lang_code_to_id[tokenizer.src_lang]

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

    # Get model instance
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'final_model_{args.annotation_mode}.pt')
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
                   name=get_wandb_exp_name(args),
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Annotation: {args.annotation_mode}"])

    del checkpoint

    # Test - Start evaluation
    model = model.eval()
    result_list = []
    ref_list = []
    hyp_list = []

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get data from batch
        source_text = data_dicts['source_text']
        target_text = data_dicts['target_text']

        model_inputs = tokenizer(source_text, text_target=None,
                                 padding='max_length', truncation=True,
                                 max_length=args.max_seq_len, return_tensors='pt')
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(**model_inputs, forced_bos_token_id=lang_id)
        generated_target_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Convert ' .' to '.' in reference - We need this trust me
        each_reference = [each_ref.replace(' .', '.') for each_ref in target_text]

        result_list.append({
            'source_text': source_text[0],
            'target_generated': generated_target_text[0],
            'target_reference': each_reference,
        })

        ref_list.append(each_reference)
        hyp_list.append(generated_target_text[0])

    # Test - nlg-eval
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    Eval = NLGEval(metrics_to_omit=['CIDEr', 'SPICE', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
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

    # Save data as json file
    save_path = os.path.join(args.result_path, args.task, args.task_dataset)
    check_path(save_path)

    result_dict = {
        'args': vars(args),
        'Bleu_1': metrics_dict['Bleu_1'],
        'Bleu_2': metrics_dict['Bleu_2'],
        'Bleu_3': metrics_dict['Bleu_3'],
        'Bleu_4': metrics_dict['Bleu_4'],
        'Bleu_avg': (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4,
        'Rouge_L': metrics_dict['ROUGE_L'],
        'Meteor': metrics_dict['METEOR'],
        'BERTScore_Precision': bert_score_P,
        'BERTScore_Recall': bert_score_R,
        'BERTScore_F1': bert_score_F1,
        'BARTScore': bart_score_total,
        'result_list': result_list,
    }
    save_name = os.path.join(save_path, f'test_result_{args.annotation_mode}_{args.learning_rate}_{args.batch_size}.json')
    with open(save_name, 'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    if args.use_tensorboard:
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
        wandb.save(save_name)

        wandb.finish()

    return metrics_dict