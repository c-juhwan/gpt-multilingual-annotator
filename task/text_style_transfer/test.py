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
from model.text_style_transfer.dataset import TSTDataset, collate_fn
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
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
    if args.annotation_mode in ['original_fr', 'translated_fr', 'gpt_fr']:
        dataset_test = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_FR.pkl'), 'test')
    elif args.annotation_mode in ['original_pt', 'translated_pt', 'gpt_pt']:
        dataset_test = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_PT.pkl'), 'test')
    elif args.annotation_mode in ['original_it', 'translated_it', 'gpt_it']:
        dataset_test = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_IT.pkl'), 'test')
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
    result_list = []
    ref_list = []
    hyp_list = []

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Train - Get data from batch
        informal_text = data_dicts['informal_text']
        all_references = data_dicts['all_references'][0]

        model_inputs = tokenizer(informal_text, text_target=None,
                                 padding='max_length', truncation=True,
                                 max_length=args.max_seq_len, return_tensors='pt')
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(**model_inputs, forced_bos_token_id=lang_id)
        generated_formal_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Convert ' .' to '.' in reference - We need this trust me
        each_reference = [each_ref.replace(' .', '.') for each_ref in all_references]

        result_list.append({
            'informal_input': informal_text[0],
            'formal_generated': generated_formal_text[0],
            'formal_reference': each_reference,
        })

        ref_list.append(each_reference)
        hyp_list.append(generated_formal_text[0])

    # Test - nlg-eval
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    Eval = NLGEval(metrics_to_omit=['CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])

    # I don't know why but we need this
    _strip = lambda x: x.strip()
    ref_list2 = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    metrics_dict = Eval.compute_metrics(ref_list2, hyp_list)
    print(metrics_dict)

    # Final - End of testing
    write_log(logger, f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
    write_log(logger, f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
    write_log(logger, f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
    write_log(logger, f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
    write_log(logger, f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
    write_log(logger, f"TEST - Rouge_L: {metrics_dict['ROUGE_L']:.4f}")
    write_log(logger, f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")

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
            'Meteor': [metrics_dict['METEOR']]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
        wandb.save(save_name)

        wandb.finish()

    return metrics_dict
