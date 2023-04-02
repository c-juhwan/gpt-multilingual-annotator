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

        for idx in tqdm(range(len(data_['input_ids'])), desc=f'Loading data from {data_path}'):
            # Load image and convert to tensor
            image_name = data_['image_names'][idx]
            image_path = os.path.join(resized_image_path, image_name)
            image = Image.open(image_path)
            image = self.transform(image)

            # Load original caption and encoded input_ids
            caption = data_['captions'][idx] # single string
            input_ids = data_['input_ids'][idx]
            #all_caption = data_['all_captions'][idx]

            self.data_list.append({
                'image': image,
                'caption': caption,
                #'all_caption': all_caption,
                'input_ids': input_ids,
                'index': idx
            })

        del data_

    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    images = torch.stack([d['image'] for d in data], dim=0) # (batch_size, 3, 224, 224)
    captions = [d['caption'] for d in data] # list of strings (batch_size)
    #all_captions = [d['all_caption'] for d in data] # list of list of strings (batch_size) - each list contains 5 captions
    input_ids = torch.stack([d['input_ids'] for d in data], dim=0) # (batch_size, max_seq_len)
    indices = [d['index'] for d in data] # list of integers

    datas_dict = {
        'image': images,
        'caption': captions,
        #'all_caption': all_captions,
        'input_ids': input_ids,
        'index': indices
    }

    return datas_dict
