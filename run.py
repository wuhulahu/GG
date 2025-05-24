import traceback

import torch
import numpy as np
import os
import random
import argparse
import GG, GG_cut_token
import datetime
from loguru import logger
from data.data_loader import load_data
from utils import tools


def run(args):
    selected_transforms = [args.soft, args.cut, args.wipe]
    if sum(selected_transforms) > 1:
        print("Error: Only one of --soft, --cut, or --erase can be selected at a time.")
        exit()
    if args.compare_flag is not False and (args.soft or args.cut or args.wipe):
        print('Error: Invalid combination of arguments for train_transform')
        exit()
    # 获取当前时间
    current_time = datetime.datetime.now()
    current_month = current_time.month
    current_day = current_time.day
    current_hour = current_time.hour
    current_minute = current_time.minute
    formatted_mark_date = "{:02d}_{:02d}".format(current_month, current_day)
    formatted_time = "{:02d}_{:02d}".format(current_hour, current_minute)
    logs_path = "logs/{}".format(formatted_mark_date)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    logger.add(
        os.path.join(logs_path, '{}_{}_{}_model_{}_codelength_{}_mu_{}_nu_{}_eta_{}_topk_{}_complement_{}.log'.format(
            args.mark,
            formatted_time,
            args.dataset,
            args.arch,
            ','.join([str(c) for c in args.code_length]),
            args.mu,
            args.nu,
            args.eta,
            args.topk,
            args.complement
        )),
        rotation='500 MB',
        level='INFO',
    )
    logger.info(args)

    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataloader, query_dataloader, retrieval_dataloader = load_data(args)

    # Training
    for code_length in args.code_length:
        logger.info('[code length:{}]'.format(code_length))
        if args.cut:
            checkpoint = GG_cut_token.train(
                train_dataloader,
                query_dataloader,
                retrieval_dataloader,
                args.arch,
                code_length,
                args.device,
                args.opt_type,
                args.lr,
                args.max_iter,
                args.mu,
                args.nu,
                args.eta,
                args.topk,
                args.evaluate_interval,
                args
            )
        else:
            checkpoint = GG.train(
                train_dataloader,
                query_dataloader,
                retrieval_dataloader,
                args.arch,
                code_length,
                args.device,
                args.opt_type,
                args.lr,
                args.max_iter,
                args.mu,
                args.nu,
                args.eta,
                args.topk,
                args.evaluate_interval,
                args
            )
        save_path = args.save_path
        torch.save(checkpoint, os.path.join(save_path, '{}_{}_{}_model_{}_codelength_{}_topk_{}_map_{:.4f}.pt'.format(args.mark, formatted_time, args.dataset, args.arch, code_length, checkpoint['map'])))
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, checkpoint['map']))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='GG_PyTorch')
    # review mark
    parser.add_argument('--mark', default='debug', type=str)
    # parser.add_argument('--debug', default=False, type=str)

    parser.add_argument('--criterion', default='DSDHLoss', type=str)
    parser.add_argument('--cl_loss', default=True, type=bool,
                        help='using Classification loss')
    # triple_loss
    parser.add_argument('--triplet', default=True, type=bool,
                        help='choose of triple_loss')
    parser.add_argument('--quadruplet', default=False, type=bool,
                        help='choose of triple_loss')
    parser.add_argument('--query_center', default=False, type=bool,
                        help='use a query to store the prior point')

    parser.add_argument('--gamma', default=0.005, type=float,
                        help='weight of triplet loss.(default: 0.005)')
    parser.add_argument('--margin', default=0.5, type=float,
                        help='margin of triple_loss.')
    parser.add_argument('--remove_hard_pos', default='not_remove_pos_max', type=str,
                        help='a addition for tri_loss')

    # setting of hyperparamsx
    parser.add_argument('--dataset', default='imagenet',
                        help="['cifar-10', 'nus-wide-tc21', 'imagenet', 'coco']")
    parser.add_argument('--erasing', default=-1,
                        help='Mask the query for robustness validation.(default: -1，0.2，0.4，0.6，0.8)')
    parser.add_argument('--soft', default=True)
    parser.add_argument('--wipe', default=False,)
    parser.add_argument('--compare_flag', default=False, help="[False, 'no_aug', 'erasing']")

    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    # note model
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='[ViTHashing, resnet50,  crossvit_small_224_cut_debug]')
    parser.add_argument('--use_custom_activation', default=False,
                        help='my activation for adjust score')
    parser.add_argument('--ratio_att', default=True,
                        help='sum the att of mask instead of area')
    parser.add_argument('--cut', default=False,
                        help='cut for convnets')
    parser.add_argument('--no_q', default=False,
                        help="For Ablation test")
    parser.add_argument('--synchronized', default=False, type=int,
                        help='synchronize the two channels(default: False)')
    parser.add_argument('--weight_remain', default=0.5, type=int,
                        help='num_remain_tokens * limit(default: 0.75)')
    parser.add_argument('--soft_per_example', default=True, type=int,
                        help='sample_soft_or_not(default: True)')
    parser.add_argument('--sigma', default=0.5, type=int,
                        help='num_remain_tokens * sigma(default: 3)')
    parser.add_argument('--limit', default=0.5, type=int,
                        help='num_remain_tokens * limit(default: 0.5)')
    parser.add_argument('--save_path', default='../../HDD/checkpoints/GG', type=str,
                        help="['checkpoints','../../HDD/checkpoints/GG']")
    # save_path = 'checkpoints'

    parser.add_argument('--pretrain', default=True)
    parser.add_argument('--opt_type', default='RMSprop', type=str,
                        help='[SGD, RMSprop]')
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate.(default: 1e-5)')
    parser.add_argument('--code_length', default='16,32,48,64', type=str,
                        help='Binary hash code length.(default: 16,32,48,64)')
    parser.add_argument('--max_iter', default=150, type=int,
                        help='Number of iterations.(default: 150)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-train', default=5000, type=int,
                        help='Number of training data points.(default: 5000)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--mu', default=1e-2, type=float,
                        help='Hyper-parameter.(default: 1e-2)')
    parser.add_argument('--nu', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--eta', default=1e-2, type=float,
                        help='Hyper-parameter.(default: 1e-2)')
    parser.add_argument('--evaluate-interval', default=10, type=int,
                        help='Evaluation interval.(default: 10)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')
    args = parser.parse_args()

    # dataset_path & Calculate map of top k
    if args.dataset == 'cifar-10':
        args.root = 'datasets/cifar-10'
        args.topk = -1
        args.num_class = 10
        args.sigma_crop = 60
    elif args.dataset == 'nus-wide-tc21':
        args.root = 'datasets/NUS-WIDE'
        args.topk = 1000
        args.num_class = 21
        args.sigma_crop = 60
    elif args.dataset == 'imagenet':
        args.root = 'datasets/Imagenet100'
        args.topk = 1000
        args.num_class = 100
        args.sigma_crop = 80
    elif args.dataset == 'coco':
        args.root = 'datasets/coco'
        args.topk = 1000
        args.num_class = 80
        args.sigma_crop = 50

    # Hash code length
    args.complement = '{}_{}_{}_{}'.format(args.dataset, args.arch, args.criterion, args.remove_hard_pos)
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == '__main__':

    args = load_config()
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
    if args.arch in ['resnet50', 'ViTHashing', 'ViTHashing_cut']:
        args.img_size = 224
    elif args.arch == 'googlenetv3':
        args.img_size = 299
    else:
        args.img_size = 240

    if 'cut' not in args.arch:
        args.layer_cut = -1
    try:
        run(args)
        tools.push_notification(args.mark)
    except Exception as e:
        traceback.print_exc()
        error_type = type(e).__name__
        tools.push_notification(args.mark, error_type=error_type)
        exit()

