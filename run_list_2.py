import copy
import itertools
import time
import torch
import numpy as np
import os
import random
import argparse
import GG, GG_cut_token
import datetime
from utils.tools import check_idle_gpu_memory, push_notification
from loguru import logger
from data.data_loader import load_data
from run import run, load_config
import traceback
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # dataset = ['cifar-10', 'nus-wide-tc21', 'imagenet', 'coco']
    dataset = ['cifar-10']

    arch = ['ViTHashing']
    # arch = ['resnet50', 'googlenetv3', 'crossvit_small_224_cut_debug', 'ViTHashing']

    # args_list = list(itertools.product(sigma_crop, arch, dataset))
    args_list = list(itertools.product(dataset, arch))

    start_time = time.time()

    for i, args_combination in enumerate(args_list):
        args = load_config()
        args.gpu = 0
        args.sigma = 0.5
        args.mark = 'debug_' + str(i + 2)
        args.dataset, args.arch = args_combination
        args.cl_loss = False
        args.soft = True
        args.cut = False
        # 消融
        args.no_q = False
        # 对比
        # [False, 'no_aug', 'other']
        args.compare_flag = False
        if 'ViTHashing' in args.arch:
            args.lr = 3e-6
        args.use_custom_activation = False
        args.ratio_att = True

        if args.arch == 'resnet50':
            args.sigma_crop = 70
        if args.arch == 'googlenetv3':
            args.sigma_crop = 80
        if args.arch == 'crossvit_small_224_cut_debug' or args.arch == 'ViTHashing':
            args.sigma_crop = 60

        # 修正path_dataset
        if args.dataset == 'cifar-10':
            args.root = 'datasets/cifar-10'
            args.topk = -1
            args.num_class = 10
            # args.max_iter = 150
        elif args.dataset == 'nus-wide-tc21':
            args.root = 'datasets/NUS-WIDE'
            args.topk = 1000
            args.num_class = 21
            # args.max_iter = 150
        elif args.dataset == 'imagenet':
            args.root = 'datasets/Imagenet100'
            args.topk = 1000
            args.num_class = 100
            # args.max_iter = 50
        elif args.dataset == 'coco':
            args.root = 'datasets/coco'
            args.topk = 1000
            args.num_class = 80

        if args.arch in ['resnet50', 'ViTHashing', 'ViTHashing_cut']:
            args.img_size = 224
        elif args.arch == 'googlenetv3':
            args.img_size = 299
        else:
            args.img_size = 240
        if 'cut' not in args.arch:
            args.layer_cut = -1

        # GPU
        if args.gpu is None:
            args.device = torch.device("cpu")
        else:
            args.device = torch.device("cuda:%d" % args.gpu)


        try:
            run(args)
        except Exception as e:
            traceback.print_exc()
            error_type = type(e).__name__
            push_notification(args.mark, error_type=error_type)
            exit()

    end_time = time.time()
    elapsed_time = end_time - start_time
    delta_time = datetime.timedelta(seconds=elapsed_time)
    days = delta_time.days
    hours, remainder = divmod(delta_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"运行时间：{days} 天, {hours} 小时, {minutes} 分钟, {seconds} 秒")
    push_notification(args.mark)