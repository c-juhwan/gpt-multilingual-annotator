# Standard Library Modules
import os
import pickle
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from PIL import Image
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class CaptioningDataset(Dataset):
    def __init__(self, args: argparse.Namespace, data_path: str, split: str) -> None:
        super(CaptioningDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.tokenizer = data_['tokenizer']

        """
        https://pytorch.org/vision/stable/models.html
        Every pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
        where H and W are expected to be at least 224.
        The images have to be loaded in to a range of [0, 1]
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        You can use the following transform to normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        """
        self.transform = transforms.Compose([
            transforms.RandomCrop(args.image_crop_size), # Crop 224x224 from 256x256 image
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),  # Normalize with predefined mean & std following torchvision documents
                                (0.229, 0.224, 0.225))
        ])

        resized_image_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, f'{split}_resized_images')

        already_loaded_data = []

        for idx in tqdm(range(len(data_['input_ids'])), desc=f'Loading data from {data_path}'):
            # Load image and convert to tensor
            image_name = data_['image_names'][idx]
            image_path = os.path.join(resized_image_path, image_name)

            if args.job == 'testing':
                if image_name in already_loaded_data:
                    continue # Skip already loaded image
                else:
                    already_loaded_data.append(image_name)

            # Load original caption and encoded input_ids
            caption = data_['captions'][idx] # single string
            input_ids = data_['input_ids'][idx]
            cap_number = data_['caption_numbers'][idx]

            self.data_list.append({
                'image_path': image_path,
                'caption': caption,
                'input_ids': input_ids,
                'index': idx,
                'caption_number': cap_number
            })

        del data_

    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    image_path = [d['image_path'] for d in data] # list of strings (batch_size)
    captions = [d['caption'] for d in data] # list of strings (batch_size)
    input_ids = torch.stack([d['input_ids'] for d in data], dim=0) # (batch_size, max_seq_len)
    indices = [d['index'] for d in data] # list of integers

    datas_dict = {
        'image_path': image_path,
        'caption': captions,
        'input_ids': input_ids,
        'index': indices
    }

    return datas_dict
