# Standard Library Modules
import os
import sys
import pickle
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from pycocotools.coco import COCO
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
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base') # Use BART tokenizer for captioning

    # Define data_dict
    data_dict = {
        'train': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
        },
        'valid': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
        },
        'test': {
            'image_names': [],
            'captions': [],
            'all_captions': [],
            'caption_numbers': [],
            'input_ids': [],
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
        tokenized_caption = tokenizer(caption, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt')

        # Append the data to the data_dict
        data_dict[split]['image_names'].append(image_name)
        data_dict[split]['captions'].append(caption)
        data_dict[split]['all_captions'].append(all_caption) # list of string
        data_dict[split]['caption_numbers'].append(caption_number)
        data_dict[split]['input_ids'].append(tokenized_caption['input_ids'].squeeze())

    # Save the data_dict for each split as pickle file
    for split in data_dict.keys():
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

    # Resize the images
    for split in data_dict.keys():
        # create the directory to store the resized images
        resized_image_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, f'{split}_resized_images')
        check_path(resized_image_path)

        if args.task_dataset == 'flickr8k':
            original_image_path = os.path.join(args.data_path, 'flickr8k', 'Images')
        elif args.task_dataset == 'flickr30k':
            original_image_path = os.path.join(args.data_path, 'flickr30k', 'flickr30k_images')
        elif args.task_dataset == 'coco2017':
            if split == 'train':
                original_image_path = os.path.join(args.data_path, 'coco', 'train2017')
            elif split == 'valid':
                original_image_path = os.path.join(args.data_path, 'coco', 'val2017')
            elif split == 'test':
                original_image_path = os.path.join(args.data_path, 'coco', 'test2017')

        # Resize the images
        for image_name in tqdm(data_dict[split]['image_names'], desc=f'Resizing {split} images...'):
            image = Image.open(os.path.join(original_image_path, image_name))
            image = image.resize((args.image_resize_size, args.image_resize_size), Image.ANTIALIAS)
            image.save(os.path.join(resized_image_path, image_name), image.format)

def get_dataset_path(args: argparse.Namespace) -> tuple: # (str, str/dict)
    # Specify the path to the dataset
    if args.task_dataset == 'flickr8k':
        # From https://www.kaggle.com/datasets/kunalgupta2616/flickr-8k-images-with-captions
        dataset_path = os.path.join(args.data_path, 'flickr8k')
    elif args.task_dataset == 'flickr30k':
        # From https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
        dataset_path = os.path.join(args.data_path, 'flickr30k')
    elif args.task_dataset == 'coco2017':
        dataset_path = os.path.join(args.data_path, 'coco')

    # Specify the path to the annotations
    if args.task_dataset == 'flickr8k':
        # From https://www.kaggle.com/datasets/kunalgupta2616/flickr-8k-images-with-captions
        annotation_path = os.path.join(dataset_path, 'captions.txt')
    elif args.task_dataset == 'flickr30k':
        # From https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
        annotation_path = os.path.join(dataset_path, 'results.csv')
    elif args.task_dataset == 'coco2017':
        annotation_path = {
            'train': os.path.join(dataset_path, 'annotations/captions_train2017.json'),
            'valid': os.path.join(dataset_path, 'annotations/captions_val2017.json')
        }
    else:
        raise ValueError('Invalid dataset name.')

    return dataset_path, annotation_path

def load_caption_data(args: argparse.Namespace) -> pd.DataFrame:
    dataset_path, annotation_path = get_dataset_path(args)

    # Load the annotations
    if args.task_dataset == 'flickr8k':
        caption_df = pd.read_csv(annotation_path, delimiter=',') # Load the captions
        caption_df = caption_df.dropna() # Drop the rows with NaN values
        caption_df = caption_df.reset_index(drop=True) # Reset the index
        caption_df = caption_df.rename(columns={'image': 'image_name', 'caption': 'caption_text'}) # Rename the columns

        # add caption number column to the dataframe
        caption_df['caption_number'] = caption_df.groupby('image_name').cumcount() + 1

        # add train/valid/test split column to the dataframe, 0 = train, 1 = valid, 2 = test
        caption_df['split'] = -1 # -1 = not assigned
        # split the dataset into train/valid/test, 80% = train, 10% = valid, 10% = test
        caption_df = caption_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        caption_df.loc[:int(len(caption_df) * 0.8), 'split'] = 0 # 80% = train
        caption_df.loc[int(len(caption_df) * 0.8):int(len(caption_df) * 0.9), 'split'] = 1 # 10% = valid
        caption_df.loc[int(len(caption_df) * 0.9):, 'split'] = 2 # 10% = test

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

        # add train/valid/test split column to the dataframe, 0 = train, 1 = valid, 2 = test
        caption_df['split'] = -1 # -1 = not assigned
        # split the dataset into train/valid/test, 80% = train, 10% = valid, 10% = test
        caption_df = caption_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        caption_df.loc[:int(len(caption_df) * 0.8), 'split'] = 0 # 80% = train
        caption_df.loc[int(len(caption_df) * 0.8):int(len(caption_df) * 0.9), 'split'] = 1 # 10% = valid
        caption_df.loc[int(len(caption_df) * 0.9):, 'split'] = 2 # 10% = test
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

    print(caption_df)
    return caption_df