import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torchvision.transforms import RandomErasing
import torchvision.transforms.functional as F
import numpy as np
import random
import math


def encode_onehot(labels, num_classes=10):
    """
    one-hot labels

    Args:
        labels (numpy.ndarray): labels.
        num_classes (int): Number of classes.

    Returns:
        onehot_labels (numpy.ndarray): one-hot labels.
    """
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels


def train_transform(img_size=240):
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        # note vit是240，其他是224
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def train_transform_compare(img_size=240, flag=False):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    compose_list = []
    if flag == 'other':
        compose_list.append(v2.ScaleJitter((img_size, img_size)))
        compose_list.append(transforms.RandomResizedCrop(img_size))
        compose_list.append(transforms.RandomHorizontalFlip())
        compose_list.append(transforms.RandomRotation(degrees=120))
        # compose_list.append(transforms.RandomGrayscale(p=0.2))

    else:
        compose_list.append(transforms.Resize((img_size, img_size)))


    compose_list.append(transforms.ToTensor())
    # if flag == 'ohter':
    #     compose_list.append(v2.ScaleJitter((img_size, img_size)))
    compose_list.append(normalize)

    return transforms.Compose(compose_list)


def train_transform_soft(img_size=240, num_class=10,max_p=1.0,
                         sigma_crop=50, pow_crop=4.0, bg_crop=1.0,
                         iou=False):
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([
        # note vit是240，其他是224
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    t_custom = SoftCrop(
        n_class=num_class,
        sigma_crop=sigma_crop, t_crop=1.0,
        max_p_crop=max_p, pow_crop=pow_crop,
        bg_crop=bg_crop,
        iou=iou)
    return t, t_custom


def train_transform_wipe(sigma=0.5, img_size=240):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([
        # note vit是240，其他是224
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])

    t_custom = CustomRandomErasing(scale=(0.02, sigma))     # scale=(0.02, 0.33)
    return t, t_custom


def query_transform(img_size=240, erasing=-1):
    """
    Query images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_list = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ]
    if erasing != -1:
        transforms_list.append(RandomErasing(p=1, scale=(erasing, erasing), ratio=(1, 1)))
    return transforms.Compose(transforms_list)


def transform_visual(img_size=240, norm=False):
    """
    Query images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_list = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if norm:
        transforms_list.append(normalize)
    return transforms.Compose(transforms_list)


def train_transform_dismatch(img_size=240, use_normalize=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transforms_list = [
        # note vit是240，其他是224
        transforms.RandomResizedCrop(img_size),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        normalize
    ]

    return transforms.Compose(transforms_list)

def query_transform_dismatch(img_size=240, use_normalize=True):
    """
    Query images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_list = [
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]

    if use_normalize:
        transforms_list.append(normalize)

    return transforms.Compose(transforms_list)


class SoftCrop:
    '''
    crop image

    '''

    def __init__(self, n_class=10,
                 sigma_crop=80, t_crop=1.0, max_p_crop=1.0, pow_crop=4.0, bg_crop=1.0,
                 iou=False):

        self.n_class = n_class
        self.chance = 1 / n_class

        # crop parameters
        self.sigma_crop = sigma_crop
        self.t_crop = t_crop
        self.max_p_crop = max_p_crop
        self.pow_crop = pow_crop
        self.bg_crop = bg_crop

        self.iou = iou  # if true, use IoU to compute r, else use IoForeground
        # for debugging
        self.flag = True

        print("use soft crop")
        print("sigma: ", self.sigma_crop, " T: ", self.t_crop, " Max P: ", self.max_p_crop,
              "bg: ", self.bg_crop, "power: ", self.pow_crop, "IoU: ", self.iou)

    def draw_offset(self, sigma=50, limit=180, n=100):
        # draw an integer from gaussian within +/- limit
        for d in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def __call__(self, image, label):

        dim1 = image.size(1)
        dim2 = image.size(2)

        # Soft Crop
        # bg = torch.randn((3,dim1*3,dim2*3)) * self.bg_crop # create a 3x by 3x sized noise background
        bg = torch.ones((3, dim1 * 3, dim2 * 3)) * self.bg_crop * torch.randn(
            (3, 1, 1))  # create a 3x by 3x sized noise background
        bg[:, dim1:2 * dim1, dim2:2 * dim2] = image  # put image at the center patch
        offset1 = self.draw_offset(self.sigma_crop, dim1)
        offset2 = self.draw_offset(self.sigma_crop, dim2)

        left = offset1 + dim1
        top = offset2 + dim2
        right = offset1 + dim1 * 2
        bottom = offset2 + dim2 * 2

        # number of pixels in orignal image kept after cropping alone
        intersection = (dim1 - abs(offset1)) * (dim2 - abs(offset2))
        # proportion of original pixels left after cutout and cropping
        if self.iou:
            overlap = intersection / (dim1 * dim2 * 2 - intersection)
        else:
            overlap = intersection / (dim1 * dim2)

        new_image = bg[:, left: right, top: bottom]  # crop image

        return new_image, overlap


def ComputeProb(x, T=0.25, n_classes=10, max_prob=1.0, pow=2.0):
    max_prob = torch.clamp_min(torch.tensor(max_prob),1/n_classes)
    if T <=0:
        T = 1e-10

    if x > T:
        return max_prob
    elif x > 0:
        a = (max_prob - 1/float(n_classes))/(T**pow)
        return max_prob - a * (T-x) ** pow
    else:
        return np.ones_like(x) * 1/n_classes


class CustomRandomErasing(RandomErasing):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        # 调用父类的构造函数，保持默认参数不变
        super().__init__(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)
        self.sigma = scale[-1]

    def forward(self, img, label):
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            erased_area_ratio = (h * w) / (img.shape[-2] * img.shape[-1])
            if erased_area_ratio >= self.sigma / 2:
                return F.erase(img, x, y, h, w, v, self.inplace), 1 - erased_area_ratio
            else:
                return F.erase(img, x, y, h, w, v, self.inplace), 1.
        return img, 1.
