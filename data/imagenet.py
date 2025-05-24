import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
from data.transform import (encode_onehot, train_transform, query_transform, train_transform_soft, train_transform_compare,
                            train_transform_wipe, transform_visual)


def load_data(root, batch_size, img_size, workers, num_class=100, soft=False, sigma_crop=60, wipe=False, sigma=0.5,
              erasing=-1, only_query=False, compare_flag=False):

    # Construct data loader
    train_dir = os.path.join(root, 'train')
    query_dir = os.path.join(root, 'query')
    retrieval_dir = os.path.join(root, 'database')

    query_dataset = ImagenetDataset(
        query_dir,
        transform=query_transform(img_size, erasing),
    )
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

    # train
    if soft:
        t, t_soft = train_transform_soft(img_size, num_class, sigma_crop=sigma_crop)
        train_dataset = ImagenetDataset(
            train_dir,
            transform=t,
            custom_transform=t_soft)
    elif wipe:
        t, t_wipe = train_transform_wipe(sigma=sigma, img_size=img_size)
        train_dataset = ImagenetDataset(
            train_dir,
            transform=t,
            custom_transform=t_wipe)
    elif compare_flag is not False:
        train_dataset = ImagenetDataset(
            train_dir,
            transform=train_transform_compare(img_size, compare_flag),
        )
    else:
        train_dataset = ImagenetDataset(
            train_dir,
            transform=train_transform(img_size),
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2
    )


    # retrieval
    retrieval_dataset = ImagenetDataset(
        retrieval_dir,
        transform=query_transform(img_size),
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return train_dataloader, query_dataloader, retrieval_dataloader



def load_data_visual(root):
    query_dir = os.path.join(root, 'query')
    retrieval_dir = os.path.join(root, 'database')

    query_dataset = ImagenetDataset(
        query_dir,
    )
    # retrieval
    retrieval_dataset = ImagenetDataset(
        retrieval_dir,
    )

    return query_dataset, retrieval_dataset


def load_data_visual_train(root, image_size):
    train_dir = os.path.join(root, 'train')
    train_dataset_norm = ImagenetDataset(
        train_dir,
        transform_visual(image_size, norm=True)
    )
    train_dataset_visual = ImagenetDataset(
        train_dir,
        transform_visual(image_size, norm=False)
    )

    return train_dataset_norm, train_dataset_visual


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, target_transform=None, custom_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.custom_transform = custom_transform
        self.imgs = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            cur_class = os.path.join(self.root, cl)
            files = os.listdir(cur_class)
            files = [os.path.join(cur_class, i) for i in files]
            self.imgs.extend(files)
            self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        self.targets = torch.tensor(self.targets)
        self.onehot_targets = torch.from_numpy(encode_onehot(self.targets, 100)).float()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img, target = self.imgs[item], self.onehot_targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.custom_transform is not None:
            img, weight = self.custom_transform(img, target)
            return img, [target, weight], item
        else:
            return img, [target], item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return self.onehot_targets

