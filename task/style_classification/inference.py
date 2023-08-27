# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import pickle
import logging
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
# Pytorch Modules
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import MBartForSequenceClassification, AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.style_classification.dataset import ClassificationDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

def inference(args: argparse.Namespace) -> None:
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

    # Preprocess generated data for inference
    pass # TODO

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset...")
    if args.task_dataset == 'xformal_fr':
        dataset_test = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_INFERENCE_FR.pkl'), 'test')
    elif args.task_dataset == 'xformal_pt':
        dataset_test = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_INFERENCE_PT.pkl'), 'test')
    elif args.task_dataset == 'xformal_it':
        dataset_test = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_INFERENCE_IT.pkl'), 'test')

    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    tokenizer = dataset_test.tokenizer

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

    # Get model instance
    model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-50", num_labels=2)

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'final_model.pt')
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
                   tags=["INFERENCE",
                         f"Dataset: {args.task_dataset}",
                         f"Annotation: {args.annotation_mode}"])

    del checkpoint

    # Inference - Start evaluation
    model = model.eval()
    test_input_list = []
    test_logit_list = []
    test_acc_cls = 0
    test_f1_cls = 0
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Inference')):
        # Inference - Get data from batch
        text = data_dicts['text']
        label = data_dicts['label'].to(device)

        model_inputs = tokenizer(text, text_target=None,
                                 padding='max_length', truncation=True,
                                 max_length=args.max_seq_len, return_tensors='pt')
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # Inference - Forward pass
        with torch.no_grad():
            outputs = model(**model_inputs, labels=None) # labels=None for inference

        # Inference - Calculate accuracy
        batch_acc_cls = (outputs.logits.argmax(dim=-1) == label).float().mean().item()
        batch_f1_cls = f1_score(label.cpu().numpy(), outputs.logits.argmax(dim=-1).cpu().numpy(), average='macro')

        # Logit: We want to save the logits of formality class - label 1
        batch_logits = outputs.logits[:, 1].cpu().numpy().tolist()
        test_logit_list.extend(batch_logits)
        test_input_list.extend(text)

        # Inference - Logging
        test_acc_cls += batch_acc_cls
        test_f1_cls += batch_f1_cls

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Acc: {batch_acc_cls:.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - F1: {batch_f1_cls:.4f}")

    # Inference - Check loss
    test_acc_cls /= len(dataloader_test)
    test_f1_cls /= len(dataloader_test)
    test_average_logit = sum(test_logit_list) / len(test_logit_list) # Average of logits

    # Inference - Save the result as json file
    result_list = []
    for each_input, each_logit in zip(test_input_list, test_logit_list):
        result_list.append({
            'input': each_input,
            'logit': each_logit,
        })
    save_path = os.path.join(args.result_path, args.task, args.task_dataset)
    check_path(save_path)
    save_name = os.path.join(save_path, f'test_result_{args.annotation_mode}_{args.learning_rate}_{args.batch_size}.json')
    with open(save_name, 'w') as f:
        json.dump({'result_list': result_list}, f, indent=4, ensure_ascii=False)

    # Final - End of inference
    write_log(logger, f"Done! - TEST - Acc: {test_acc_cls:.4f} - F1: {test_f1_cls:.4f} - Formality Logit: {test_average_logit:.4f}")
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Acc': [test_acc_cls],
            'F1': [test_f1_cls],
            'Average_Formality_Logit': [test_average_logit],
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'INFERENCE_Result': wandb_table})
        wandb.save(save_name)

        wandb.finish()

    return test_acc_cls, test_f1_cls

def preprocess_generated_data(args: argparse.Namespace) -> None:
    # open the generated json file
    save_path = os.path.join(args.result_path, 'text_style_transfer', args.task_dataset)
    with open(os.path.join(save_path, f'test_result_{args.annotation_mode}_{args.learning_rate}_{args.batch_size}.json'), 'r') as f:
        raw_data = json.load(f)
    result_list = raw_data['result_list']

    inference_data_dict = {
        'text': [],
        'label': [],
        'category': [],
        'tokenizer': None,
    }

    # Load data to inference_data_dict
    for each_result in result_list:
        inference_data_dict['text'].append(each_result['formal_generated'])
        inference_data_dict['label'].append(None) # label is None for inference
        inference_data_dict['category'].append('fr') # Family & Relationships

    # Get tokenizer
    if args.task_dataset == 'xformal_fr':
        lang_code = 'fr_XX'
        out_lang_code = 'FR'
    elif args.task_dataset == 'xformal_pt':
        lang_code = 'pt_XX'
        out_lang_code = 'PT'
    elif args.task_dataset == 'xformal_it':
        lang_code = 'it_IT'
        out_lang_code = 'IT'

    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50", src_lang=lang_code, tgt_lang=lang_code)
    inference_data_dict['tokenizer'] = tokenizer

    # Save the data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)

    with open(os.path.join(preprocessed_path, f'test_INFERENCE_{out_lang_code}.pkl'), 'wb') as f:
        pickle.dump(inference_data_dict, f)
