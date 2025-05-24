import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from data.transform import train_transform, query_transform, encode_onehot, train_transform_soft, \
    train_transform_dismatch, query_transform_dismatch, train_transform_compare, train_transform_wipe, transform_visual
import torchvision.transforms as transforms


def load_data(root, num_query, num_train, batch_size, img_size, num_workers, num_class, soft=False, sigma_crop=80,
              wipe=False, sigma=0.5, erasing=-1, only_query=False, compare_flag=False):
    """
    Load cifar10 dataset.

    Args
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    CIFAR10.init(root, num_query, num_train)
    # query
    query_dataset = CIFAR10('query', transform=query_transform(img_size, erasing))
    query_dataloader = DataLoader(
        query_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )
    # 当验证erasing效果的时候，某些情况只需要返回query_dataloader
    if only_query:
        return None, query_dataloader, None

    # train
    if soft:
        t, t_soft = train_transform_soft(img_size, num_class, sigma_crop=sigma_crop)
        train_dataset = CIFAR10('train', transform=t, target_transform=None, custom_transform=t_soft)
    elif wipe:
        t, t_wipe = train_transform_wipe(sigma=sigma, img_size=img_size)
        train_dataset = CIFAR10('train', transform=t, target_transform=None, custom_transform=t_wipe)
    elif compare_flag is not False:
        train_dataset = CIFAR10('train', transform=train_transform_compare(img_size, compare_flag), target_transform=None)
    else:
        train_dataset = CIFAR10('train', transform=train_transform(img_size), target_transform=None)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
      )

    # retrieval
    retrieval_dataset = CIFAR10('database', transform=query_transform(img_size))
    # retrieval_dataset = CIFAR10('database', transform=train_transform_dismatch(img_size))
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )

    return train_dataloader, query_dataloader, retrieval_dataloader


def load_data_dismatch(root, num_query, num_train, batch_size, img_size, num_workers, use_normalize=True):
    """
    Load cifar10 dataset.

    Args
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    CIFAR10.init(root, num_query, num_train, toy_dismatch=True)

    # train
    train_dataset = CIFAR10('train', transform=train_transform_dismatch(img_size), target_transform=None)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
      )

    # query,retrieval dataset
    # train_test
    query_dataset = CIFAR10('query', transform=query_transform_dismatch(img_size, use_normalize=use_normalize))

    # retrieval
    retrieval_dataset = CIFAR10('database', transform=query_transform_dismatch(img_size, use_normalize=use_normalize))

    # query,retrieval dataloader
    query_dataloader = DataLoader(
        query_dataset,
        shuffle=False,
        batch_size=batch_size,
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



def load_data_visual(root, num_query, num_train, toy_dismatch=False):
    # 没有使用transformers 所以不需要特地去掉normalization
    CIFAR10.init(root, num_query, num_train, toy_dismatch=toy_dismatch)
    # query
    query_dataset = CIFAR10('query')
    # retrieval
    retrieval_dataset = CIFAR10('database')
    return query_dataset, retrieval_dataset


def load_data_visual_train(root, image_size):
    CIFAR10.init(root, 1000, 5000,)
    return CIFAR10('train', transform=transform_visual(image_size, norm=True)), CIFAR10('train',transform=transform_visual(image_size, norm=False))


class CIFAR10(Dataset):
    """
    Cifar10 dataset.
    """
    @staticmethod
    def init(root, num_query, num_train, toy_dismatch=False):
        data_list = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5',
                     'test_batch',
                     ]
        # base_folder = 'cifar-10-batches-py'

        data = []
        targets = []

        for file_name in data_list:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)

        # Sort by class
        sort_index = targets.argsort()
        data = data[sort_index, :]
        targets = targets[sort_index]
        CIFAR10.DATA = data

        # (num_query / number of class) query images per class
        # (num_train / number of class) train images per class
        query_per_class = num_query // 10
        train_per_class = num_train // 10

        # Permutate index (range 0 - 6000 per class)
        perm_index = np.random.permutation(data.shape[0] // 10)
        query_index = perm_index[:query_per_class]
        train_index = perm_index[query_per_class: query_per_class + train_per_class]

        query_index = np.tile(query_index, 10)
        train_index = np.tile(train_index, 10)
        inc_index = np.array([i * (data.shape[0] // 10) for i in range(10)])
        query_index = query_index + inc_index.repeat(query_per_class)
        train_index = train_index + inc_index.repeat(train_per_class)
        list_query_index = [i for i in query_index]
        # retrieval_index = np.array(list(set(range(data.shape[0])) - set(list_query_index)), dtype=int)
        # todo
        retrieval_index = np.array(list(set(range(data.shape[0])) - set(list_query_index) - set(train_index)), dtype=int)

        # Split data, targets
        CIFAR10.QUERY_IMG = data[query_index, :]
        CIFAR10.QUERY_TARGET = targets[query_index]
        CIFAR10.TRAIN_IMG = data[train_index, :]
        CIFAR10.TRAIN_TARGET = targets[train_index]
        CIFAR10.RETRIEVAL_IMG = data[retrieval_index, :]
        CIFAR10.RETRIEVAL_TARGET = targets[retrieval_index]

        if toy_dismatch:
            retrieval_rotation_img = (np.load('datasets/cifar-10/images_rotation.npy')*255).astype(np.uint8).transpose((0, 2, 3, 1))
            retrieval_rotation_tatget = np.load('datasets/cifar-10/labels_rotation.npy')
            CIFAR10.RETRIEVAL_IMG = np.vstack((retrieval_rotation_img, CIFAR10.RETRIEVAL_IMG))
            CIFAR10.RETRIEVAL_TARGET = np.concatenate((retrieval_rotation_tatget.argmax(1), CIFAR10.RETRIEVAL_TARGET))


    def __init__(self, mode='train',
                 transform=None, target_transform=None, custom_transform=None
                 ):
        self.transform = transform
        self.target_transform = target_transform
        self.custom_transform = custom_transform

        if mode == 'train':
            self.data = CIFAR10.TRAIN_IMG
            self.targets = CIFAR10.TRAIN_TARGET
        elif mode == 'query':
            self.data = CIFAR10.QUERY_IMG
            self.targets = CIFAR10.QUERY_TARGET
        else:
            self.data = CIFAR10.RETRIEVAL_IMG
            self.targets = CIFAR10.RETRIEVAL_TARGET

        self.onehot_targets = encode_onehot(self.targets, 10)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.onehot_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.custom_transform is not None:
            img, weight = self.custom_transform(img, target)
            return img, [target, weight], index
        else:
            return img, [target], index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.onehot_targets).float()


