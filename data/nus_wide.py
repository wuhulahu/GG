import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform, train_transform_soft, train_transform_compare, train_transform_wipe

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(tc, root, num_query, num_train, batch_size, img_size, num_workers,
              num_class=100, soft=False, sigma_crop=60, wipe=False, sigma=0.5, erasing=-1, only_query=False, compare_flag=False):
    """
    Loading nus-wide dataset.

    Args:
        tc(int): Top class.
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    if soft:
        t, t_custom = train_transform_soft(img_size, num_class, sigma_crop=sigma_crop)
    elif wipe:
        t, t_custom = train_transform_wipe(sigma=sigma, img_size=img_size)
    elif compare_flag is not False:
        t = train_transform_compare(img_size, compare_flag)
        t_custom = None
    else:
        t = train_transform(img_size)
        t_custom = None

    if tc == 21:
        query_dataset = NusWideDatasetTC21(
            root,
            'test_img.txt',
            'test_label_onehot.txt',
            transform=query_transform(img_size, erasing),
        )

        train_dataset = NusWideDatasetTC21(
            root,
            'database_img.txt',
            'database_label_onehot.txt',
            transform=t,
            train=True,
            num_train=num_train,
            custom_transform=t_custom
        )

        retrieval_dataset = NusWideDatasetTC21(
            root,
            'database_img.txt',
            'database_label_onehot.txt',
            transform=query_transform(img_size),
        )
    elif tc == 10:
        NusWideDatasetTc10.init(root, num_query, num_train)
        query_dataset = NusWideDatasetTc10(root, 'query', query_transform(img_size, erasing))
        train_dataset = NusWideDatasetTc10(root, 'train', train_transform(img_size))
        retrieval_dataset = NusWideDatasetTc10(root, 'retrieval', query_transform(img_size))

    query_dataloader = DataLoader(
        query_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )

    return train_dataloader, query_dataloader, retrieval_dataloader


class NusWideDatasetTc10(Dataset):
    """
    Nus-wide dataset, 10 classes.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = NusWideDatasetTc10.TRAIN_DATA
            self.targets = NusWideDatasetTc10.TRAIN_TARGETS
        elif mode == 'query':
            self.data = NusWideDatasetTc10.QUERY_DATA
            self.targets = NusWideDatasetTc10.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = NusWideDatasetTc10.RETRIEVAL_DATA
            self.targets = NusWideDatasetTc10.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index

    def __len__(self):
        return self.data.shape[0]

    def get_targets(self):
        return torch.from_numpy(self.targets).float()

    @staticmethod
    def init(root, num_query, num_train):
        """
        Initialize dataset.

        Args
            root(str): Path of image files.
            num_query(int): Number of query data.
            num_train(int): Number of training data.
        """
        # Load dataset
        img_txt_path = os.path.join(root, 'img_tc10.txt')
        targets_txt_path = os.path.join(root, 'targets_onehot_tc10.txt')

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=np.int64)

        # Split dataset
        perm_index = np.random.permutation(data.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query:]

        NusWideDatasetTc10.QUERY_DATA = data[query_index]
        NusWideDatasetTc10.QUERY_TARGETS = targets[query_index, :]

        NusWideDatasetTc10.TRAIN_DATA = data[train_index]
        NusWideDatasetTc10.TRAIN_TARGETS = targets[train_index, :]

        NusWideDatasetTc10.RETRIEVAL_DATA = data[retrieval_index]
        NusWideDatasetTc10.RETRIEVAL_TARGETS = targets[retrieval_index, :]


class NusWideDatasetTC21(Dataset):
    """
    Nus-wide dataset, 21 classes.

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name.
        label_txt(str): Path of txt file containing image label.
        transform(callable, optional): Transform images.
        train(bool, optional): Return training dataset.
        num_train(int, optional): Number of training data.
    """
    def __init__(self, root, img_txt, label_txt, transform=None, train=None, num_train=None, custom_transform=None):
        self.root = root
        self.transform = transform
        self.custom_transform = custom_transform

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)

        # Read files
        with open(img_txt_path, 'r') as f:
            self.data = np.array([i.strip() for i in f])
        self.targets = np.loadtxt(label_txt_path, dtype=np.float32)

        # Sample training dataset
        if train is True:
            perm_index = np.random.permutation(len(self.data))[:num_train]
            self.data = self.data[perm_index]
            self.targets = self.targets[perm_index]

    # def __getitem__(self, index):
    #     img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     return img, self.targets[index], index

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.custom_transform is not None:
            img, weight = self.custom_transform(img, self.targets[index])
            return img, [self.targets[index], weight], index
        else:
            return img, [self.targets[index]], index


    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()
