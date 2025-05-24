import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import (encode_onehot, train_transform, query_transform, train_transform_soft, train_transform_compare,
                            train_transform_wipe)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, batch_size, img_size, workers, num_class=80, soft=False, sigma_crop=60, wipe=False, sigma=0.5,
              erasing=-1, only_query=False, compare_flag=False):
    """
    Loading nus-wide dataset.

    Args:
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader (torch.evaluate.data.DataLoader): Data loader.
    """

    # COCO.init(root, num_query, num_train)
    query_dataset = COCO(root, 'query', query_transform(img_size, erasing))

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2
    )
    if only_query:
        return None, query_dataloader, None

    if soft:
        t, t_soft = train_transform_soft(img_size, num_class, sigma_crop=sigma_crop)
        train_dataset = COCO(root, 'train', transform=t, custom_transform=t_soft)
    elif wipe:
        t, t_wipe = train_transform_wipe(sigma=sigma, img_size=img_size)
        train_dataset = COCO(root, 'train', transform=t, custom_transform=t_wipe)
    elif compare_flag is not False:
        train_dataset = COCO(
            root,
            'train',
            transform=train_transform_compare(img_size, compare_flag),
        )
    else:
        train_dataset = COCO(
            root,
            'train',
            transform=train_transform(img_size),
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=workers,
        prefetch_factor=2
    )

    retrieval_dataset = COCO(root, 'database', query_transform(img_size))
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return train_dataloader, query_dataloader, retrieval_dataloader


class COCO(Dataset):
    """
    COCO dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None, target_transform=None, custom_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.custom_transform = custom_transform

        # Load dataset
        dataset_info_txt_path = root + '/' + mode + '.txt'


        with open(dataset_info_txt_path, "r") as file:
            data_targets = [line.strip().split(" ") for line in file]

        self.imgs = np.array(['images/' + item[0] for item in data_targets], dtype=str)
        self.targets = np.array([list(map(int, item[1:])) for item in data_targets], dtype=np.int64)

    def __getitem__(self, item):
        img, target = self.imgs[item], self.targets[item]

        img = Image.open(os.path.join(self.root, self.imgs[item])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.custom_transform is not None:
            img, weight = self.custom_transform(img, target)
            return img, [target, weight], item
        else:
            return img, [target], item

    def __len__(self):
        return self.imgs.shape[0]

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()
