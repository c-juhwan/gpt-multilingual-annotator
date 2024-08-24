# Standard Library Modules
import os
import argparse
# Custom Modules
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.user_name = os.getlogin()
        self.proj_name = 'Captioning_GPT'

        # Task arguments
        task_list = ['captioning', 'annotating',
                     'text_style_transfer', 'annotating_tst', 'style_classification',
                     'machine_translation', 'annotating_mt']
        self.parser.add_argument('--task', type=str, choices=task_list, default='captioning',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing', 'eval_similarity', 'inference', # For captioning/TST/classification
                    'gpt_annotating', 'backtrans_annotating', 'eda_annotating', 'synonym_annotating', 'onlyone_annotating', 'budget_annotating', 'translation_annotating', 'googletrans_annotating'] # For annotating
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['flickr8k', 'flickr30k', 'coco2014', 'coco2017',
                        'uit_viic', 'aide', 'new_lv', 'new_et', 'new_fi',
                        'gyafc_en', 'xformal_fr', 'xformal_pt', 'xformal_it',
                        'multi30k']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='flickr8k',
                                 help='Dataset for the task; Must be given.')
        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')
        annotation_mode_list = ['original_en', 'aihub_ko', 'gpt_en', 'gpt_ko', # Korean dataset AIHub
                                'backtrans_en', 'eda_en', 'synonym_en', 'onlyone_en', 'hrqvae_en', 'budget_en', # Main experiment
                                'coco_en', 'original_vie', 'translated_vie', 'gpt_vie', 'hrqtrans_vie', 'googletrans_vie', # Vietnamese dataset UIT-ViIC
                                'original_pl', 'translated_pl', 'gpt_pl', 'hrqtrans_pl', 'googletrans_pl', # Polish dataset AIDe
                                'translated_lv', 'gpt_lv', 'hrqtrans_lv', 'googletrans_lv', # Dataset construction: Latvian
                                'translated_et', 'gpt_et', 'hrqtrans_et', 'googletrans_et', # Dataset construction: Estonian
                                'translated_fi', 'gpt_fi', 'hrqtrans_fi', 'googletrans_fi', # Dataset construction: Finnish
                                'original_fr', 'translated_fr', 'gpt_fr', 'googletrans_fr', # French dataset XFormal
                                'original_pt', 'translated_pt', 'gpt_pt', 'googletrans_pt', # Portuguese dataset XFormal
                                'original_it', 'translated_it', 'gpt_it', 'googletrans_it', # Italian dataset XFormal
                                'original_de', 'translated_de', 'gpt_de', 'googletrans_de',] # German dataset Multi30k
        self.parser.add_argument('--annotation_mode', type=str, choices=annotation_mode_list, default='original_en',
                                 help='Annotation mode; Default is "original"')

        # Path arguments
        self.parser.add_argument('--data_path', type=str, default='/mnt/sda1/dataset',
                                 help='Path to the raw dataset before preprocessing.')
        self.parser.add_argument('--preprocess_path', type=str, default=f'/mnt/sda1/{self.user_name}/preprocessed',
                                 help='Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'/mnt/sda1/{self.user_name}/model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'/mnt/sda1/{self.user_name}/model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type=str, default=f'/mnt/sda1/{self.user_name}/results/{self.proj_name}',
                                 help='Path to the result after testing.')
        self.parser.add_argument('--log_path', type=str, default=f'/mnt/sda1/{self.user_name}/tensorboard_log/{self.proj_name}',
                                 help='Path to the tensorboard log file.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default=self.proj_name,
                                 help='Name of the project for tensorboard.')
        encoder_type_list = ['resnet50', 'efficientnet_b0', 'vit_b_16']
        self.parser.add_argument('--encoder_type', default='vit_b_16', choices=encoder_type_list, type=str,
                                 help='Encoder type; Default is vit_b_16')
        self.parser.add_argument('--encoder_pretrained', default=True, type=bool,
                                 help='Whether to use pretrained encoder; Default is True')
        decoder_type_list = ['transformer', 'lstm']
        self.parser.add_argument('--decoder_type', default='transformer', choices=decoder_type_list, type=str,
                                 help='Decoder type; Default is transformer')
        self.parser.add_argument('--min_seq_len', type=int, default=4,
                                 help='Minimum sequence length of the output; Default is 4')
        self.parser.add_argument('--max_seq_len', type=int, default=100,
                                 help='Maximum sequence length of the output; Default is 100')
        self.parser.add_argument('--dropout_rate', type=float, default=0.2,
                                 help='Dropout rate of the model; Default is 0.2')

        # Model - Size arguments
        self.parser.add_argument('--embed_size', type=int, default=768,
                                 help='Embedding size of the model; Default is 768')
        self.parser.add_argument('--hidden_size', type=int, default=768,
                                 help='Hidden size of the model; Default is 768')
        self.parser.add_argument('--decoder_transformer_nhead', default=12, type=int,
                                 help='Decoder nhead for decoder_type==transformer; Default is 12')
        self.parser.add_argument('--decoder_transformer_nlayers', default=12, type=int,
                                 help='Decoder number of layers for decoder_type==transformer; Default is 12')
        self.parser.add_argument('--decoder_lstm_nlayers', default=2, type=int,
                                 help='Decoder number of layers for decoder_type==lstm; Default is 2')

        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='AdamW',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='CosineAnnealingLR',
                                 help="Scheduler to use for classification; If None, no scheduler is used; Default is CosineAnnealingLR")

        # Training arguments 1
        self.parser.add_argument('--num_epochs', type=int, default=5,
                                 help='Training epochs; Default is 5')
        self.parser.add_argument('--learning_rate', type=float, default=5e-5,
                                 help='Learning rate of optimizer; Default is 5e-5')
        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help='Batch size; Default is 16')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5,
                                 help='Weight decay; Default is 5e-4; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=5,
                                 help='Gradient clipping norm; Default is 5')
        self.parser.add_argument('--label_smoothing_eps', type=float, default=0.05,
                                 help='Label smoothing epsilon; Default is 0.05')
        self.parser.add_argument('--early_stopping_patience', type=int, default=10,
                                 help='Early stopping patience; Default is 10 -> No early stopping')
        objective_list = ['loss', 'accuracy']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='accuracy',
                                 help='Objective to optimize; Default is accuracy')

        # Preprocessing - Image preprocessing config
        self.parser.add_argument('--image_resize_size', default=256, type=int,
                                 help='Size of resized image after preprocessing.')
        self.parser.add_argument('--image_crop_size', default=224, type=int,
                                 help='Size of cropped image after preprocessing.')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=1, type=int,
                                 help='Batch size for test; Default is 1')
        strategy_list = ['greedy', 'beam', 'multinomial', 'topk', 'topp']
        self.parser.add_argument('--decoding_strategy', type=str, choices=strategy_list, default='greedy',
                                 help='Decoding strategy for test; Default is greedy')
        self.parser.add_argument('--beam_size', default=5, type=int,
                                 help='Beam search size; Default is 5')
        self.parser.add_argument('--beam_alpha', default=0.7, type=float,
                                 help='Beam search length normalization; Default is 0.7')
        self.parser.add_argument('--beam_repetition_penalty', default=1.3, type=float,
                                 help='Beam search repetition penalty term; Default is 1.3')
        self.parser.add_argument('--topk', default=5, type=int,
                                 help='Topk sampling size; Default is 5')
        self.parser.add_argument('--topp', default=0.9, type=float,
                                 help='Topk sampling size; Default is 0.9')
        self.parser.add_argument('--softmax_temp', default=1.0, type=float,
                                 help='Softmax temperature; Default is 1.0')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda:0',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--gpt_model_version', type=str, default='gpt-3.5-turbo',
                                 help='GPT version to use for annotating; Default is gpt-3.5-turbo')
        self.parser.add_argument('--seed', default=None,
                                 help='Random seed; Default is None;')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=False,
                                 help='Using tensorboard; Default is False')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb; Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()