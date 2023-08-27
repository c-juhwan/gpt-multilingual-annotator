# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import shutil
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
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

def training(args: argparse.Namespace) -> None:
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

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset...")
    dataset_dict, dataloader_dict = {}, {}
    if args.annotation_mode == 'original_fr': # XFormal Original (French)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_ORIGINAL_FR.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_FR.pkl'), 'valid')
    elif args.annotation_mode == 'translated_fr': # NLLB Translated (GYAFC->French)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_TRANSLATED_FR.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_FR.pkl'), 'valid')
    elif args.annotation_mode == 'gpt_fr' and args.gpt_model_version == 'gpt-4': # GPT Annotated (French)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_GPT4_FR.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_FR.pkl'), 'valid')
    elif args.annotation_mode == 'original_pt': # XFormal Original (Brazilian Portuguese)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_ORIGINAL_PT.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_PT.pkl'), 'valid')
    elif args.annotation_mode == 'translated_pt': # NLLB Translated (GYAFC->Brazilian Portuguese)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_TRANSLATED_PT.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_PT.pkl'), 'valid')
    elif args.annotation_mode == 'gpt_pt' and args.gpt_model_version == 'gpt-4': # GPT Annotated (Brazilian Portuguese)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_GPT4_PT.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_PT.pkl'), 'valid')
    elif args.annotation_mode == 'original_it': # XFormal Original (Italian)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_ORIGINAL_IT.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_IT.pkl'), 'valid')
    elif args.annotation_mode == 'translated_it': # NLLB Translated (GYAFC->Italian)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_TRANSLATED_IT.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_IT.pkl'), 'valid')
    elif args.annotation_mode == 'gpt_it' and args.gpt_model_version == 'gpt-4': # GPT Annotated (Italian)
        dataset_dict['train'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'train_GPT4_IT.pkl'), 'train')
        dataset_dict['valid'] = TSTDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, 'valid_ORIGINAL_IT.pkl'), 'valid')
    else:
        raise ValueError(f'Invalid annotation mode: {args.annotation_mode}')

    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    tokenizer = dataset_dict['train'].tokenizer

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")
    write_log(logger, f"Valid dataset size / iterations: {len(dataset_dict['valid'])} / {len(dataloader_dict['valid'])}")

    # Get model instance
    write_log(logger, "Building model")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)

    # Define optimizer and scheduler
    write_log(logger, "Building optimizer and scheduler")
    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, optim_type=args.optimizer)
    scheduler = get_scheduler(optimizer, len(dataloader_dict['train']), num_epochs=args.num_epochs,
                              early_stopping_patience=args.early_stopping_patience, learning_rate=args.learning_rate,
                              scheduler_type=args.scheduler)
    write_log(logger, f"Optimizer: {optimizer}")
    write_log(logger, f"Scheduler: {scheduler}")

    # We don't need explicit loss function - it is already implemented in the model

    # If resume_training, load from checkpoint
    start_epoch = 0
    if args.job == 'resume_training':
        write_log(logger, "Resuming training model")
        load_checkpoint_name = os.path.join(args.checkpoint_path, args.task, args.task_dataset,
                                            f'{args.encoder_type}_{args.decoder_type}_checkpoint_{args.annotation_mode}.pt')
        model = model.to('cpu')
        checkpoint = torch.load(load_checkpoint_name, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.to(device)
        write_log(logger, f"Loaded checkpoint from {load_checkpoint_name}")

        if args.use_wandb:
            import wandb # Only import wandb when it is used
            from wandb import AlertLevel
            wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Annotation: {args.annotation_mode}",
                             f"Encoder: {args.encoder_type}",
                             f"Decoder: {args.decoder_type}"],
                       resume=True,
                       id=checkpoint['wandb_id'])
            # wandb.watch(models=model, criterion=seq_loss, log='all', log_freq=10)
        del checkpoint

    # Initialize tensorboard writer
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Initialize wandb
    if args.use_wandb and args.job == 'training':
        import wandb # Only import wandb when it is used
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Annotation: {args.annotation_mode}",
                             f"Encoder: {args.encoder_type}",
                             f"Decoder: {args.decoder_type}"])
        # wandb.watch(models=model, criterion=seq_loss, log='all', log_freq=10)

    # Train/Valid - Start training
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    write_log(logger, f"Start training from epoch {start_epoch}")
    for epoch_idx in range(start_epoch, args.num_epochs):
        # Train - Set model to train mode
        model = model.train()
        train_loss_seq = 0

        # Train - Iterate one epoch over batches
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]')):
            # Train - Get data from batch
            informal_text = data_dicts['informal_text']
            formal_text = data_dicts['formal_text']

            model_inputs = tokenizer(informal_text, text_target=formal_text,
                                     padding='max_length', truncation=True,
                                     max_length=args.max_seq_len, return_tensors='pt')
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # Train - Forward pass
            outputs = model(**model_inputs, return_dict=True)

            # Train - Calculate loss
            batch_loss_seq = outputs.loss
            train_loss_seq += batch_loss_seq.item()

            # Train - Backward pass
            optimizer.zero_grad()
            batch_loss_seq.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step() # These schedulers require step() after every training iteration

            # Train - Logging
            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['train']) - 1:
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Loss: {batch_loss_seq.item():.4f}")
            if args.use_tensorboard:
                writer.add_scalar('TRAIN/Learning_Rate', optimizer.param_groups[0]['lr'], epoch_idx * len(dataloader_dict['train']) + iter_idx)

        # Train - End of epoch logging
        if args.use_tensorboard:
            writer.add_scalar('TRAIN/Loss', train_loss_seq / len(dataloader_dict['train']), epoch_idx)

        # Valid - Set model to eval mode
        model = model.eval()
        valid_loss_seq = 0

        # Valid - Iterate one epoch over batches
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc=f'Validation - Epoch [{epoch_idx}/{args.num_epochs}]')):
            # Valid - Get data from batch
            informal_text = data_dicts['informal_text']
            formal_text = data_dicts['formal_text']

            model_inputs = tokenizer(informal_text, text_target=formal_text,
                                     padding='max_length', truncation=True,
                                     max_length=args.max_seq_len, return_tensors='pt')
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}


            # Valid - Forward pass
            with torch.no_grad():
                outputs = model(**model_inputs, return_dict=True)

            # Valid - Calculate loss
            batch_loss_seq = outputs.loss
            valid_loss_seq += batch_loss_seq.item()

            # Valid - Logging
            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['valid']) - 1:
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Loss: {batch_loss_seq.item():.4f}")

        # Valid - Call scheduler
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_loss_seq)

        # Valid - Check loss & save model
        valid_loss_seq /= len(dataloader_dict['valid'])

        if args.optimize_objective == 'loss':
            valid_objective_value = valid_loss_seq
            valid_objective_value = -1 * valid_objective_value # Loss is minimized, but we want to maximize the objective value
        else:
            raise NotImplementedError

        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"VALID - Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0 # Reset early stopping counter

            checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset)
            check_path(checkpoint_save_path)

            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'wandb_id': wandb.run.id if args.use_wandb else ''
            }, os.path.join(checkpoint_save_path, f'checkpoint_{args.annotation_mode}.pt'))
            write_log(logger, f"VALID - Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
            write_log(logger, f"VALID - Saved checkpoint to {checkpoint_save_path}")
        else:
            early_stopping_counter += 1
            write_log(logger, f"VALID - Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")

        # Valid - End of epoch logging
        if args.use_tensorboard:
            writer.add_scalar('VALID/Loss', valid_loss_seq, epoch_idx)
        if args.use_wandb:
            wandb.log({'TRAIN/Epoch_Loss': train_loss_seq / len(dataloader_dict['train']),
                       'VALID/Epoch_Loss': valid_loss_seq,
                       'Epoch_Index': epoch_idx})
            wandb.alert(
                title='Epoch End',
                text=f"VALID - Epoch {epoch_idx} - Loss: {valid_loss_seq:.4f}",
                level=AlertLevel.INFO,
                wait_duration=300
            )

        # Valid - Early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f"VALID - Early stopping at epoch {epoch_idx}...")
            break

    # Final - End of training
    write_log(logger, f"Done! Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
    if args.use_tensorboard:
        writer.add_text('VALID/Best', f"Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
        writer.close()

    # Final - Save best checkpoint as result model
    final_model_save_path = os.path.join(args.model_path, args.task, args.task_dataset)
    check_path(final_model_save_path)
    shutil.copyfile(os.path.join(checkpoint_save_path, f'checkpoint_{args.annotation_mode}.pt'),
                    os.path.join(final_model_save_path, f'final_model_{args.annotation_mode}.pt')) # Copy best checkpoint as final model
    write_log(logger, f"FINAL - Saved final model to {final_model_save_path}")

    if args.use_wandb:
        wandb.finish()