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
from sklearn.metrics import f1_score
# Pytorch Modules
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import MBartForSequenceClassification
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.style_classification.dataset import ClassificationDataset, collate_fn
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
    if args.task_dataset == 'xformal_fr':
        dataset_test = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_FR.pkl'), 'test')
    elif args.task_dataset == 'xformal_pt':
        dataset_test = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_PT.pkl'), 'test')
    elif args.task_dataset == 'xformal_it':
        dataset_test = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'test_ORIGINAL_IT.pkl'), 'test')

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
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Annotation: {args.annotation_mode}"])

    del checkpoint

    # Test - Start evaluation
    model = model.eval()
    test_loss_cls = 0
    test_acc_cls = 0
    test_f1_cls = 0
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get data from batch
        text = data_dicts['text']
        label = data_dicts['label'].to(device)

        model_inputs = tokenizer(text, text_target=None,
                                     padding='max_length', truncation=True,
                                     max_length=args.max_seq_len, return_tensors='pt')
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # Test - Forward pass
        with torch.no_grad():
            outputs = model(**model_inputs, labels=label)

        # Test - Calculate loss and accuracy
        batch_loss_cls = outputs.loss
        batch_acc_cls = (outputs.logits.argmax(dim=-1) == label).float().mean().item()
        batch_f1_cls = f1_score(label.cpu().numpy(), outputs.logits.argmax(dim=-1).cpu().numpy(), average='macro')

        # Test - Logging
        test_loss_cls += batch_loss_cls.item()
        test_acc_cls += batch_acc_cls
        test_f1_cls += batch_f1_cls

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Loss: {batch_loss_cls.item():.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Acc: {batch_acc_cls:.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - F1: {batch_f1_cls:.4f}")

    # Test - Check loss
    test_loss_cls /= len(dataloader_test)
    test_acc_cls /= len(dataloader_test)
    test_f1_cls /= len(dataloader_test)

    # Final - End of testing
    write_log(logger, f"Done! - TEST - Loss: {test_loss_cls:.4f} - Acc: {test_acc_cls:.4f} - F1: {test_f1_cls:.4f}")
    if args.use_tensorboard:
        writer.add_scalar('TEST/Loss', test_loss_cls, 0)
        writer.add_scalar('TEST/Acc', test_acc_cls, 0)
        writer.add_scalar('TEST/F1', test_f1_cls, 0)
        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Acc': [test_acc_cls],
            'F1': [test_f1_cls],
            'Loss': [test_loss_cls]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'TEST_Result': wandb_table})

        wandb.finish()

    return test_acc_cls, test_f1_cls
