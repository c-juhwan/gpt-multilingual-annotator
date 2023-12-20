# Standard Library Modules
import os
import sys
import json
import pickle
import random
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from pycocotools.coco import COCO
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.
    """
    # Load the data
    caption_df = load_caption_data(args)

    # Define tokenizer
    en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base') # Use BART tokenizer for captioning
    ko_tokenizer = AutoTokenizer.from_pretrained('cosmoquester/bart-ko-base')
    vie_tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable')
    pl_tokenizer = AutoTokenizer.from_pretrained('sdadas/polish-bart-base')
    lv_tokenizer = AutoTokenizer.from_pretrained('joelito/legal-latvian-roberta-base')
    et_tokenizer = AutoTokenizer.from_pretrained('tartuNLP/EstBERT')
    fi_tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-uncased-v1')


    # Define data_dict
    data_dict = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': en_tokenizer,
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': en_tokenizer,
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': en_tokenizer,
        },
    }
    data_dict_ko = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': ko_tokenizer,
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': ko_tokenizer,
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': ko_tokenizer,
        },
    }
    data_dict_vie = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': vie_tokenizer,
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': vie_tokenizer,
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': vie_tokenizer,
        },
    }
    data_dict_pl = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': pl_tokenizer,
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': pl_tokenizer,
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': pl_tokenizer,
        },
    }
    data_dict_lv = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': lv_tokenizer,
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': lv_tokenizer,
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': lv_tokenizer,
        },
    }
    data_dict_et = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': et_tokenizer,
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': et_tokenizer,
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': et_tokenizer,
        },
    }
    data_dict_fi = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': fi_tokenizer,
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': fi_tokenizer,
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
            'tokenizer': fi_tokenizer,
        },
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)

    for idx in tqdm(range(len(caption_df)), desc='Preprocessing captions...'):
        # Get the data from the dataframe
        image_name = caption_df['image_name'][idx]
        caption = caption_df['caption_text'][idx]
        image_all_caption_df = caption_df[caption_df['image_name'] == image_name] # find the caption with same image name
        all_caption = image_all_caption_df['caption_text'].tolist()
        caption_number = caption_df['caption_number'][idx]
        split_ = caption_df['split'][idx]
        split = 'train' if split_ == 0 else 'valid' if split_ == 1 else 'test'

        # Tokenize the caption
        tokenized_caption = en_tokenizer(caption, padding='max_length', truncation=True,
                                         max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to the data_dict
        data_dict[split]['image_names'].append(image_name)
        data_dict[split]['caption_numbers'].append(caption_number)
        data_dict[split]['captions'].append(caption)
        data_dict[split]['all_captions'].append(all_caption)
        data_dict[split]['input_ids'].append(tokenized_caption['input_ids'].squeeze())

    if args.task_dataset in ['flickr8k', 'flickr30k']:
        # For flickr, remain only 1 data for each image in valid and test split
        for split in ['valid', 'test']:
            # gather only caption_number == 1
            data_dict_new = {
                'image_names': [],
                'captions': [],
                'all_captions': [],
                'caption_numbers': [],
                'input_ids': [],
                'tokenizer': data_dict[split]['tokenizer'],
            }

            for idx in range(len(data_dict[split]['caption_numbers'])):
                if data_dict[split]['caption_numbers'][idx] == 1:
                    data_dict_new['image_names'].append(data_dict[split]['image_names'][idx])
                    data_dict_new['caption_numbers'].append(data_dict[split]['caption_numbers'][idx])
                    data_dict_new['captions'].append(data_dict[split]['captions'][idx])
                    data_dict_new['all_captions'].append(data_dict[split]['all_captions'][idx])
                    data_dict_new['input_ids'].append(data_dict[split]['input_ids'][idx])
                else:
                    continue

            data_dict[split] = data_dict_new

        print(len(data_dict['train']['image_names']))
        print(len(data_dict['valid']['image_names']))
        print(len(data_dict['test']['image_names']))

    # Save the data_dict for each split as pickle file
    for split in data_dict.keys():
        with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_EN.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

    if args.task_dataset == 'coco2014': # Process the Korean captions for the COCO2014 dataset
        for idx in tqdm(range(len(caption_df)), desc='Preprocessing Korean captions...'):
            # Get the data from the dataframe
            image_name = caption_df['image_name'][idx]
            caption = caption_df['caption_text_ko'][idx]
            image_all_caption_df = caption_df[caption_df['image_name'] == image_name] # find the caption with same image name
            all_caption = image_all_caption_df['caption_text'].tolist()
            caption_number = caption_df['caption_number'][idx]
            split_ = caption_df['split'][idx]
            split = 'train' if split_ == 0 else 'valid' if split_ == 1 else 'test'

            # Tokenize the caption
            tokenized_caption = ko_tokenizer(caption, padding='max_length', truncation=True,
                                             max_length=args.max_seq_len-1, return_tensors='pt') # -1 for [BOS], [EOS] will be added later
            tokenized_caption_ = []
            # Add [BOS] and [EOS] tokens
            for each_input_id in tokenized_caption['input_ids']:
                each_input_id = torch.cat([torch.tensor([ko_tokenizer.bos_token_id]), each_input_id])
                # Find the first [PAD] token and replace it with [EOS] token
                first_pad_idx = torch.where(each_input_id == ko_tokenizer.pad_token_id)[0][0]
                each_input_id[first_pad_idx] = ko_tokenizer.eos_token_id
                tokenized_caption_.append(each_input_id)
            tokenized_caption_ = torch.stack(tokenized_caption_) # Convert list to tensor

            # Append the data to the data_dict
            data_dict_ko[split]['image_names'].append(image_name)
            data_dict_ko[split]['caption_numbers'].append(caption_number)
            data_dict_ko[split]['captions'].append(caption)
            data_dict_ko[split]['all_captions'].append(all_caption)
            data_dict_ko[split]['input_ids'].append(tokenized_caption_.squeeze())

        # Save the data_dict for each split as pickle file
        for split in data_dict_ko.keys():
            with open(os.path.join(preprocessed_path, f'{split}_AIHUB_KO.pkl'), 'wb') as f:
                pickle.dump(data_dict_ko[split], f)
    if args.task_dataset == 'uit_viic': # Process the Vietnamese captions for the UIT-ViIC dataset
        # Change name for english processed data_dict
        for split in data_dict.keys():
            os.rename(os.path.join(preprocessed_path, f'{split}_ORIGINAL_EN.pkl'), os.path.join(preprocessed_path, f'{split}_COCO_EN.pkl'))

        for idx in tqdm(range(len(caption_df)), desc='Preprocessing Vietnamese captions...'):
            # Get the data from the dataframe
            image_name = caption_df['image_name'][idx]
            caption = caption_df['caption_text_vie'][idx]
            image_all_caption_df = caption_df[caption_df['image_name'] == image_name] # find the caption with same image name
            all_caption = image_all_caption_df['caption_text_vie'].tolist()
            # Remove None from all_caption list
            all_caption = [x for x in all_caption if x is not None]
            caption_number = caption_df['caption_number'][idx]
            split_ = caption_df['split'][idx]
            split = 'train' if split_ == 0 else 'valid' if split_ == 1 else 'test'

            if caption == None:
                continue
            # Tokenize the caption
            tokenized_caption = vie_tokenizer(caption, padding='max_length', truncation=True,
                                              max_length=args.max_seq_len, return_tensors='pt')

            # Append the data to the data_dict
            data_dict_vie[split]['image_names'].append(image_name)
            data_dict_vie[split]['caption_numbers'].append(caption_number)
            data_dict_vie[split]['captions'].append(caption)
            data_dict_vie[split]['all_captions'].append(all_caption)
            data_dict_vie[split]['input_ids'].append(tokenized_caption['input_ids'].squeeze())

        # Save the data_dict for each split as pickle file
        for split in data_dict_vie.keys():
            with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_VIE.pkl'), 'wb') as f:
                pickle.dump(data_dict_vie[split], f)
    if args.task_dataset == 'aide': # Process the Polish captions for the AIDe dataset
        # Change name for english processed data_dict
        for split in data_dict.keys():
            os.rename(os.path.join(preprocessed_path, f'{split}_ORIGINAL_EN.pkl'), os.path.join(preprocessed_path, f'{split}_FLICKR_EN.pkl'))

        for idx in tqdm(range(len(caption_df)), desc='Preprocessing Polish captions...'):
            # Get the data from the dataframe
            image_name = caption_df['image_name'][idx]
            caption = caption_df['caption_text_pl'][idx]
            image_all_caption_df = caption_df[caption_df['image_name'] == image_name] # find the caption with same image name
            all_caption = image_all_caption_df['caption_text_pl'].tolist()
            split_ = caption_df['split'][idx]
            split = 'train' if split_ == 0 else 'valid' if split_ == 1 else 'test'

            # Tokenize the caption
            tokenized_caption = pl_tokenizer(caption, padding='max_length', truncation=True,
                                             max_length=args.max_seq_len, return_tensors='pt')

            # Append the data to the data_dict
            data_dict_pl[split]['image_names'].append(image_name)
            data_dict_pl[split]['caption_numbers'].append(caption_number)
            data_dict_pl[split]['captions'].append(caption)
            data_dict_pl[split]['all_captions'].append(all_caption)
            data_dict_pl[split]['input_ids'].append(tokenized_caption['input_ids'].squeeze())

        # Save the data_dict for each split as pickle file
        for split in data_dict_pl.keys():
            with open(os.path.join(preprocessed_path, f'{split}_ORIGINAL_PL.pkl'), 'wb') as f:
                pickle.dump(data_dict_pl[split], f)
    if args.task_dataset == 'new_lv': # Process the Latvian captions for the new_lv dataset
        # Change name for english processed data_dict
        # Actual dataset construction will be constructed in annotation step
        for split in data_dict.keys():
            os.rename(os.path.join(preprocessed_path, f'{split}_ORIGINAL_EN.pkl'), os.path.join(preprocessed_path, f'{split}_COCO_EN.pkl'))
    if args.task_dataset == 'new_et': # Process the Estonian captions for the new_et dataset
        # Change name for english processed data_dict
        # Actual dataset construction will be constructed in annotation step
        for split in data_dict.keys():
            os.rename(os.path.join(preprocessed_path, f'{split}_ORIGINAL_EN.pkl'), os.path.join(preprocessed_path, f'{split}_COCO_EN.pkl'))
    if args.task_dataset == 'new_fi': # Process the Finnish captions for the new_fi dataset
        # Change name for english processed data_dict
        # Actual dataset construction will be constructed in annotation step
        for split in data_dict.keys():
            os.rename(os.path.join(preprocessed_path, f'{split}_ORIGINAL_EN.pkl'), os.path.join(preprocessed_path, f'{split}_COCO_EN.pkl'))

    # Resize the images
    for split in data_dict.keys():
        # create the directory to store the resized images
        resized_image_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, f'{split}_resized_images')
        check_path(resized_image_path)

        if args.task_dataset in ['flickr8k', 'aide']:
            original_image_path = os.path.join(args.data_path, 'flickr8k', 'Images')
        elif args.task_dataset == 'flickr30k':
            original_image_path = os.path.join(args.data_path, 'flickr30k', 'flickr30k_images')
        elif args.task_dataset in ['coco2014', 'uit_viic', 'new_lv', 'new_et', 'new_fi']:
            if split == 'train':
                original_image_path = os.path.join(args.data_path, 'coco_2014', 'train2014')
            elif split == 'valid':
                original_image_path = os.path.join(args.data_path, 'coco_2014', 'val2014')
            elif split == 'test':
                original_image_path = os.path.join(args.data_path, 'coco_2014', 'test2014')
        elif args.task_dataset == 'coco2017':
            if split == 'train':
                original_image_path = os.path.join(args.data_path, 'coco', 'train2017')
            elif split == 'valid':
                original_image_path = os.path.join(args.data_path, 'coco', 'val2017')
            elif split == 'test':
                original_image_path = os.path.join(args.data_path, 'coco', 'test2017')

        # remove duplicated image names
        data_dict[split]['image_names'] = list(set(data_dict[split]['image_names']))

        # Resize the images
        for image_name in tqdm(data_dict[split]['image_names'], desc=f'Resizing {split} images...'):
            if args.task_dataset not in ['uit_viic', 'new_lv', 'new_et', 'new_fi']:
                image = Image.open(os.path.join(original_image_path, image_name)).convert('RGB') # convert to RGB if the image is grayscale
                image = image.resize((args.image_resize_size, args.image_resize_size), Image.ANTIALIAS)
                image.save(os.path.join(resized_image_path, image_name), image.format)
            elif args.task_dataset in ['uit_viic', 'new_lv', 'new_et', 'new_fi']: # for uit_viic, we will only save the resized images for the train/valid/test split
                # if image is included in the train/valid/test split, save the image
                if image_name in data_dict[split]['image_names']:
                    if "train" in image_name:
                        image = Image.open(os.path.join(args.data_path, 'coco_2014', 'train2014', image_name)).convert('RGB')
                    elif "val" in image_name:
                        image = Image.open(os.path.join(args.data_path, 'coco_2014', 'val2014', image_name)).convert('RGB')
                    image = image.resize((args.image_resize_size, args.image_resize_size), Image.ANTIALIAS)

                    image.save(os.path.join(resized_image_path, image_name), image.format)
                else:
                    continue # if not, skip the image
            elif args.task_dataset in ['aide', 'new_lv', 'new_et', 'new_fi']:
                # if image is included in the train/valid/test split, save the image
                if image_name in data_dict[split]['image_names']:
                    image = Image.open(os.path.join(original_image_path, image_name)).convert('RGB') # convert to RGB if the image is grayscale
                    image = image.resize((args.image_resize_size, args.image_resize_size), Image.ANTIALIAS)
                    image.save(os.path.join(resized_image_path, image_name), image.format)
                else:
                    continue

def get_dataset_path(args: argparse.Namespace) -> tuple: # (str, str/dict)
    # Specify the path to the dataset
    if args.task_dataset == 'flickr8k':
        # From https://www.kaggle.com/datasets/kunalgupta2616/flickr-8k-images-with-captions
        dataset_path = os.path.join(args.data_path, 'flickr8k')
    elif args.task_dataset == 'flickr30k':
        # From https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
        dataset_path = os.path.join(args.data_path, 'flickr30k')
    elif args.task_dataset == 'coco2014':
        dataset_path = os.path.join(args.data_path, 'coco_2014')
    elif args.task_dataset == 'coco2017':
        dataset_path = os.path.join(args.data_path, 'coco')
    elif args.task_dataset == 'uit_viic':
        dataset_path = os.path.join(args.data_path, 'UIT-ViIC')
    elif args.task_dataset == 'aide':
        dataset_path = os.path.join(args.data_path, 'AIDe')
    elif args.task_dataset == 'new_lv':
        dataset_path = os.path.join(f'/home/{os.getlogin()}/Workspace/image-captioning-gpt-annotator/task/captioning/new_lv_data')
    elif args.task_dataset == 'new_et':
        dataset_path = os.path.join(f'/home/{os.getlogin()}/Workspace/image-captioning-gpt-annotator/task/captioning/new_et_data')
    elif args.task_dataset == 'new_fi':
        dataset_path = os.path.join(f'/home/{os.getlogin()}/Workspace/image-captioning-gpt-annotator/task/captioning/new_fi_data')

    # Specify the path to the annotations
    if args.task_dataset == 'flickr8k':
        # From https://www.kaggle.com/datasets/kunalgupta2616/flickr-8k-images-with-captions
        annotation_path = os.path.join(dataset_path, 'captions.txt')
    elif args.task_dataset == 'flickr30k':
        # From https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
        annotation_path = os.path.join(dataset_path, 'results.csv')
    elif args.task_dataset == 'coco2014':
        # https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100
        annotation_path = {
            'train': os.path.join(dataset_path, 'annotations/captions_train2014_korean.json'), # ai-hub korean caption dataset
            'valid': os.path.join(dataset_path, 'annotations/captions_val2014_korean.json')
        }
    elif args.task_dataset == 'coco2017':
        annotation_path = {
            'train': os.path.join(dataset_path, 'annotations/captions_train2017.json'),
            'valid': os.path.join(dataset_path, 'annotations/captions_val2017.json')
        }
    elif args.task_dataset == 'uit_viic':
        annotation_path = {
            'train_vie': os.path.join(dataset_path, 'uitviic_captions_train2017.json'),
            'train_eng': os.path.join(dataset_path, 'uitviic_captions_train2017_EN.json'),
            'valid_vie': os.path.join(dataset_path, 'uitviic_captions_val2017.json'),
            'valid_eng': os.path.join(dataset_path, 'uitviic_captions_val2017_EN.json'),
            'test_vie': os.path.join(dataset_path, 'uitviic_captions_test2017.json'),
            'test_eng': os.path.join(dataset_path, 'uitviic_captions_test2017_EN.json'),
            'image_train': os.path.join(args.data_path, 'coco_2014', 'train2014'), # for image path, we use coco2014 image folder
            'image_valid': os.path.join(args.data_path, 'coco_2014', 'val2014')
        }
    elif args.task_dataset == 'aide':
        annotation_path = {
            'train': os.path.join(dataset_path, 'aide_train.json'),
            'valid': os.path.join(dataset_path, 'aide_val.json'),
            'test': os.path.join(dataset_path, 'aide_test.json'),
            'image': os.path.join(args.data_path, 'flickr8k', 'Images')
        }
    elif args.task_dataset == 'new_lv':
        annotation_path = {
            'train_eng': os.path.join(dataset_path, 'latvian_train2017_EN.json'),
            'valid_eng': os.path.join(dataset_path, 'latvian_val2017_EN.json'),
            'test_eng': os.path.join(dataset_path, 'latvian_test2017_EN.json'),
            'image_train': os.path.join(args.data_path, 'coco_2014', 'train2014'), # for image path, we use coco2014 image folder
            'image_valid': os.path.join(args.data_path, 'coco_2014', 'val2014')
        }
    elif args.task_dataset == 'new_et':
        annotation_path = {
            'train_eng': os.path.join(dataset_path, 'estonian_train2017_EN.json'),
            'valid_eng': os.path.join(dataset_path, 'estonian_val2017_EN.json'),
            'test_eng': os.path.join(dataset_path, 'estonian_test2017_EN.json'),
            'image_train': os.path.join(args.data_path, 'coco_2014', 'train2014'), # for image path, we use coco2014 image folder
            'image_valid': os.path.join(args.data_path, 'coco_2014', 'val2014')
        }
    elif args.task_dataset == 'new_fi':
        annotation_path = {
            'train_eng': os.path.join(dataset_path, 'finnish_train2017_EN.json'),
            'valid_eng': os.path.join(dataset_path, 'finnish_val2017_EN.json'),
            'test_eng': os.path.join(dataset_path, 'finnish_test2017_EN.json'),
            'image_train': os.path.join(args.data_path, 'coco_2014', 'train2014'), # for image path, we use coco2014 image folder
            'image_valid': os.path.join(args.data_path, 'coco_2014', 'val2014')
        }
    else:
        raise ValueError('Invalid dataset name.')

    return dataset_path, annotation_path

def load_caption_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.seed is not None:
        random.seed(args.seed)

    dataset_path, annotation_path = get_dataset_path(args)

    # Load the annotations
    if args.task_dataset == 'flickr8k':
        caption_df = pd.read_csv(annotation_path, delimiter=',') # Load the captions
        caption_df = caption_df.dropna() # Drop the rows with NaN values
        caption_df = caption_df.reset_index(drop=True) # Reset the index
        caption_df = caption_df.rename(columns={'image': 'image_name', 'caption': 'caption_text'}) # Rename the columns

        # add caption number column to the dataframe
        caption_df['caption_number'] = caption_df.groupby('image_name').cumcount() + 1 # Start from 1

        # add train/valid/test split column to the dataframe, 0 = train, 1 = valid, 2 = test
        caption_df['split'] = -1 # -1 = not assigned
        # split the dataset into train/valid/test, 80% = train, 10% = valid, 10% = test
        # gather the image names
        image_names = list(set(caption_df['image_name']))
        random.shuffle(image_names)
        # split the image names into train/valid/test
        train_image_names = image_names[:int(len(image_names) * 0.8)]
        valid_image_names = image_names[int(len(image_names) * 0.8):int(len(image_names) * 0.9)]
        test_image_names = image_names[int(len(image_names) * 0.9):]

        # assign the split to the dataframe
        caption_df.loc[caption_df['image_name'].isin(train_image_names), 'split'] = 0
        caption_df.loc[caption_df['image_name'].isin(valid_image_names), 'split'] = 1
        caption_df.loc[caption_df['image_name'].isin(test_image_names), 'split'] = 2

        # preprocess the captions: remove " from the captions
        # some captions have " in the captions, which causes misunderstanding to the model
        caption_df['caption_text'] = caption_df['caption_text'].apply(lambda x: x.replace('"', ''))
    elif args.task_dataset == 'flickr30k':
        caption_df = pd.read_csv(annotation_path, delimiter='|') # Load the captions

        # change column names -> remove the spaces
        caption_df.columns = ['image_name', 'comment_number', 'comment']
        # print any row with nan value
        print(caption_df[caption_df.isna().any(axis=1)])
        # as there is a flaw in the csv file, we need to manually fix it.
        # row 19999 has no separator, so we need to manually add it.
        caption_df.loc[19999, 'image_name'] = '2199200615.jpg'
        caption_df.loc[19999, 'comment_number'] = 4
        caption_df.loc[19999, 'comment'] = 'A dog runs across the grass .'
        caption_df = caption_df.dropna() # Drop the rows with NaN values
        caption_df = caption_df.reset_index(drop=True) # Reset the index
        caption_df = caption_df.rename(columns={'image_name': 'image_name', 'comment_number': 'caption_number', 'comment': 'caption_text'}) # Rename the columns
        caption_df['caption_number'] = caption_df.groupby('image_name').cumcount() + 1 # Start from 1

        # add train/valid/test split column to the dataframe, 0 = train, 1 = valid, 2 = test
        caption_df['split'] = -1 # -1 = not assigned
        # split the dataset into train/valid/test, 80% = train, 10% = valid, 10% = test
        # gather the image names
        image_names = list(set(caption_df['image_name']))
        random.shuffle(image_names)
        # split the image names into train/valid/test
        train_image_names = image_names[:int(len(image_names) * 0.8)]
        valid_image_names = image_names[int(len(image_names) * 0.8):int(len(image_names) * 0.9)]
        test_image_names = image_names[int(len(image_names) * 0.9):]

        # assign the split to the dataframe
        caption_df.loc[caption_df['image_name'].isin(train_image_names), 'split'] = 0
        caption_df.loc[caption_df['image_name'].isin(valid_image_names), 'split'] = 1
        caption_df.loc[caption_df['image_name'].isin(test_image_names), 'split'] = 2
    elif args.task_dataset == 'coco2014':
        train_df = pd.read_json(annotation_path['train'])
        valid_df = pd.read_json(annotation_path['valid'])

        # create empty dataframe to store the captions
        caption_df = pd.DataFrame(columns=['image_name', 'caption_number', 'caption_text', 'caption_text_ko', 'split'])

        # get the captions for the train set
        for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Loading coco train captions'):
            row = row.to_dict()['annotations']
            image_name = row['file_path']
            image_name = os.path.basename(image_name)
            captions_en = row['captions'] # List
            captions_ko = row['caption_ko'] # List
            for j, caption_en, caption_ko in zip(range(len(captions_en)), captions_en, captions_ko):
                caption_df = caption_df.append({'image_name': image_name, 'caption_number': j+1, # Start from 1
                                                'caption_text': caption_en, 'caption_text_ko': caption_ko, 'split': 0}, ignore_index=True)

        # get the captions for the valid set
        for i, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc='Loading coco valid captions'):
            row = row.to_dict()['annotations']
            image_name = row['file_path']
            image_name = os.path.basename(image_name)
            captions_en = row['captions']
            captions_ko = row['caption_ko']
            for j, caption_en, caption_ko in zip(range(len(captions_en)), captions_en, captions_ko):
                caption_df = caption_df.append({'image_name': image_name, 'caption_number': j+1, # Start from 1
                                                'caption_text': caption_en, 'caption_text_ko': caption_ko, 'split': 1}, ignore_index=True)

        # for test set, we just append the image names to the dataframe
        test_image_names = os.listdir(os.path.join(dataset_path, 'test2014'))
        for image_name in tqdm(test_image_names, desc='Loading coco test captions'):
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': 1,
                                            'caption_text': '', 'caption_text_ko': '', 'split': 2}, ignore_index=True)

        print(caption_df)
        return caption_df
    elif args.task_dataset == 'coco2017':
        train_coco = COCO(annotation_path['train'])
        valid_coco = COCO(annotation_path['valid'])

        # create empty dataframe to store the captions
        caption_df = pd.DataFrame(columns=['image_name', 'caption_number', 'caption_text', 'split'])

        # get the captions for the train set
        train_image_ids = train_coco.getImgIds()
        for image_id in tqdm(train_image_ids, desc='Loading coco train captions'):
            image_name = train_coco.loadImgs(image_id)[0]['file_name']
            caption_ids = train_coco.getAnnIds(imgIds=image_id)
            annotations = train_coco.loadAnns(caption_ids)

            for i, annotation in enumerate(annotations):
                caption_df = caption_df.append({'image_name': image_name, 'caption_number': i+1,
                                                'caption_text': annotation['caption'], 'split': 0},
                                                ignore_index=True)

        # get the captions for the valid set
        valid_image_ids = valid_coco.getImgIds()
        for image_id in tqdm(valid_image_ids, desc='Loading coco valid captions'):
            image_name = valid_coco.loadImgs(image_id)[0]['file_name']
            caption_ids = valid_coco.getAnnIds(imgIds=image_id)
            annotations = valid_coco.loadAnns(caption_ids)

            for i, annotation in enumerate(annotations):
                caption_df = caption_df.append({'image_name': image_name, 'caption_number': i+1,
                                                'caption_text': annotation['caption'], 'split': 1},
                                                ignore_index=True)

        # for test set, we just append the image names to the dataframe
        test_image_names = os.listdir(os.path.join(dataset_path, 'test2017'))
        for image_name in tqdm(test_image_names, desc='Loading coco test captions'):
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': 1,
                                            'caption_text': '', 'split': 2},
                                            ignore_index=True)
    elif args.task_dataset == 'uit_viic':
        with open(annotation_path['train_vie'], 'r') as f:
            train_vie_json = json.load(f)
            train_vie_df = pd.DataFrame(train_vie_json['annotations'])
        with open(annotation_path['train_eng'], 'r') as f:
            train_eng_json = json.load(f)
            train_eng_df = pd.DataFrame(train_eng_json['annotations'])
        with open(annotation_path['valid_vie'], 'r') as f:
            valid_vie_json = json.load(f)
            valid_vie_df = pd.DataFrame(valid_vie_json['annotations'])
        with open(annotation_path['valid_eng'], 'r') as f:
            valid_eng_json = json.load(f)
            valid_eng_df = pd.DataFrame(valid_eng_json['annotations'])
        with open(annotation_path['test_vie'], 'r') as f:
            test_vie_json = json.load(f)
            test_vie_df = pd.DataFrame(test_vie_json['annotations'])
        with open(annotation_path['test_eng'], 'r') as f:
            test_eng_json = json.load(f)
            test_eng_df = pd.DataFrame(test_eng_json['annotations'])

        # Create empty dataframe to store the captions
        caption_df = pd.DataFrame(columns=['image_name', 'caption_number', 'caption_text', 'caption_text_vie', 'split'])

        for i in tqdm(range(len(train_eng_df)), total=len(train_eng_df), desc='Loading COCO train captions'):
            image_id = train_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = train_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_vie': None, 'split': 0}, ignore_index=True)

        # get the captions for the train set
        for i in tqdm(range(len(train_vie_df)), total=len(train_vie_df), desc='Loading UIT-ViIC train captions'):
            image_id = train_vie_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_vie = train_vie_df.iloc[i]['caption'] # Str

            # Fill the caption_text_vie column of caption_df
            # Find the row with same image_name and caption_number with empty caption_text_vie
            caption_number = caption_df[(caption_df['image_name'] == image_name) & (caption_df['caption_text_vie'].isnull())]['caption_number'].min()

            # Fill the caption_text_vie column
            caption_df.loc[(caption_df['image_name'] == image_name) & (caption_df['caption_number'] == caption_number), 'caption_text_vie'] = caption_vie

        # get the captions for the valid set
        for i in tqdm(range(len(valid_eng_df)), total=len(valid_eng_df), desc='Loading COCO valid captions'):
            image_id = valid_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = valid_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_vie': None, 'split': 1}, ignore_index=True)

        # get the captions for the valid set
        for i in tqdm(range(len(valid_vie_df)), total=len(valid_vie_df), desc='Loading UIT-ViIC valid captions'):
            image_id = valid_vie_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_vie = valid_vie_df.iloc[i]['caption'] # Str

            # Fill the caption_text_vie column of caption_df
            # Find the row with same image_name and caption_number with empty caption_text_vie
            caption_number = caption_df[(caption_df['image_name'] == image_name) & (caption_df['caption_text_vie'].isnull())]['caption_number'].min()

            # Fill the caption_text_vie column
            caption_df.loc[(caption_df['image_name'] == image_name) & (caption_df['caption_number'] == caption_number), 'caption_text_vie'] = caption_vie

        # get the captions for the test set
        for i in tqdm(range(len(test_eng_df)), total=len(test_eng_df), desc='Loading COCO test captions'):
            image_id = test_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = test_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_vie': None, 'split': 2}, ignore_index=True)

        # get the captions for the test set
        for i in tqdm(range(len(test_vie_df)), total=len(test_vie_df), desc='Loading UIT-ViIC test captions'):
            image_id = test_vie_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_vie = test_vie_df.iloc[i]['caption'] # Str

            # Fill the caption_text_vie column of caption_df
            # Find the row with same image_name and caption_number with empty caption_text_vie
            caption_number = caption_df[(caption_df['image_name'] == image_name) & (caption_df['caption_text_vie'].isnull())]['caption_number'].min()

            # Fill the caption_text_vie column
            caption_df.loc[(caption_df['image_name'] == image_name) & (caption_df['caption_number'] == caption_number), 'caption_text_vie'] = caption_vie

        print(caption_df)
        return caption_df
    elif args.task_dataset == 'new_lv':
        # Actual dataset construction will be conducted in annotation process
        with open(annotation_path['train_eng'], 'r') as f:
            train_eng_json = json.load(f)
            train_eng_df = pd.DataFrame(train_eng_json['annotations'])
        with open(annotation_path['valid_eng'], 'r') as f:
            valid_eng_json = json.load(f)
            valid_eng_df = pd.DataFrame(valid_eng_json['annotations'])
        with open(annotation_path['test_eng'], 'r') as f:
            test_eng_json = json.load(f)
            test_eng_df = pd.DataFrame(test_eng_json['annotations'])

        # Create empty dataframe to store the captions
        caption_df = pd.DataFrame(columns=['image_name', 'caption_number', 'caption_text', 'caption_text_lv', 'split'])

        for i in tqdm(range(len(train_eng_df)), total=len(train_eng_df), desc='Loading COCO train captions'):
            image_id = train_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = train_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_lv': None, 'split': 0}, ignore_index=True)

        # get the captions for the valid set
        for i in tqdm(range(len(valid_eng_df)), total=len(valid_eng_df), desc='Loading COCO valid captions'):
            image_id = valid_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = valid_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_lv': None, 'split': 1}, ignore_index=True)

        # get the captions for the test set
        for i in tqdm(range(len(test_eng_df)), total=len(test_eng_df), desc='Loading COCO test captions'):
            image_id = test_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = test_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_lv': None, 'split': 2}, ignore_index=True)

        print(caption_df)
        return caption_df
    elif args.task_dataset == 'new_et':
        # Actual dataset construction will be conducted in annotation process
        with open(annotation_path['train_eng'], 'r') as f:
            train_eng_json = json.load(f)
            train_eng_df = pd.DataFrame(train_eng_json['annotations'])
        with open(annotation_path['valid_eng'], 'r') as f:
            valid_eng_json = json.load(f)
            valid_eng_df = pd.DataFrame(valid_eng_json['annotations'])
        with open(annotation_path['test_eng'], 'r') as f:
            test_eng_json = json.load(f)
            test_eng_df = pd.DataFrame(test_eng_json['annotations'])

        # Create empty dataframe to store the captions
        caption_df = pd.DataFrame(columns=['image_name', 'caption_number', 'caption_text', 'caption_text_et', 'split'])

        for i in tqdm(range(len(train_eng_df)), total=len(train_eng_df), desc='Loading COCO train captions'):
            image_id = train_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = train_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_et': None, 'split': 0}, ignore_index=True)

        # get the captions for the valid set
        for i in tqdm(range(len(valid_eng_df)), total=len(valid_eng_df), desc='Loading COCO valid captions'):
            image_id = valid_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = valid_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_et': None, 'split': 1}, ignore_index=True)

        # get the captions for the test set
        for i in tqdm(range(len(test_eng_df)), total=len(test_eng_df), desc='Loading COCO test captions'):
            image_id = test_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = test_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_et': None, 'split': 2}, ignore_index=True)

        print(caption_df)
        return caption_df
    elif args.task_dataset == 'new_fi':
        # Actual dataset construction will be conducted in annotation process
        with open(annotation_path['train_eng'], 'r') as f:
            train_eng_json = json.load(f)
            train_eng_df = pd.DataFrame(train_eng_json['annotations'])
        with open(annotation_path['valid_eng'], 'r') as f:
            valid_eng_json = json.load(f)
            valid_eng_df = pd.DataFrame(valid_eng_json['annotations'])
        with open(annotation_path['test_eng'], 'r') as f:
            test_eng_json = json.load(f)
            test_eng_df = pd.DataFrame(test_eng_json['annotations'])

        # Create empty dataframe to store the captions
        caption_df = pd.DataFrame(columns=['image_name', 'caption_number', 'caption_text', 'caption_text_et', 'split'])

        for i in tqdm(range(len(train_eng_df)), total=len(train_eng_df), desc='Loading COCO train captions'):
            image_id = train_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = train_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_et': None, 'split': 0}, ignore_index=True)

        # get the captions for the valid set
        for i in tqdm(range(len(valid_eng_df)), total=len(valid_eng_df), desc='Loading COCO valid captions'):
            image_id = valid_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = valid_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_et': None, 'split': 1}, ignore_index=True)

        # get the captions for the test set
        for i in tqdm(range(len(test_eng_df)), total=len(test_eng_df), desc='Loading COCO test captions'):
            image_id = test_eng_df.iloc[i]['image_id']
            image_name = f'COCO_train2014_{image_id:012d}.jpg' # COCO image name format
            # Check if the image exists in the image folder
            if not os.path.exists(os.path.join(annotation_path['image_train'], image_name)):
                image_name = f'COCO_val2014_{image_id:012d}.jpg' # Go to val2014 folder
                if not os.path.exists(os.path.join(annotation_path['image_valid'], image_name)):
                    raise FileNotFoundError(f'{image_name} does not exist in the image folder.')
            caption_eng = test_eng_df.iloc[i]['caption'] # Str
            # If there is previous caption with same image name, caption_number + 1
            if len(caption_df[caption_df['image_name'] == image_name]) > 0:
                caption_number = caption_df[caption_df['image_name'] == image_name]['caption_number'].max() + 1
            else:
                caption_number = 1 # Start from 1
            caption_df = caption_df.append({'image_name': image_name, 'caption_number': caption_number,
                                            'caption_text': caption_eng, 'caption_text_et': None, 'split': 2}, ignore_index=True)

        print(caption_df)
        return caption_df
    elif args.task_dataset == 'aide':
        train_df = pd.read_json(annotation_path['train'])
        valid_df = pd.read_json(annotation_path['valid'])
        test_df = pd.read_json(annotation_path['test'])

        # Combine the train/valid/test dataframe into one dataframe
        caption_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        print(caption_df)
        return caption_df
