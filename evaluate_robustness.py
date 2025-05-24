import torch
import os
import re
import argparse
from loguru import logger
from run import load_config
from models.model_loader import load_model
from data.data_loader import load_data
from GG import mean_average_precision, generate_code
from GG_cut_token import generate_code as generate_code_cut


def create_model_log(model_name, dataset_name):
    logs_path = "logs/robustness_test/"
    return os.path.join(logs_path, f"robustness_test_{dataset_name}_{model_name}.log")


def extract_info_from_filename(filename):
    pattern = r'(.+?)_(\d+_\d+_\d+_\d+_\d+)_(.+)_model_(.+)_codelength_(\d+)_.*\.pt'
    match = re.match(pattern, filename)
    if match:
        # datetime = match.group(1)
        dataset = match.group(3)
        model = match.group(4)
        code_length = match.group(5)
        return dataset, model, int(code_length)
    else:
        return None


def load_model_and_weights(model_path, args):
    dataset, model_name, code_length = extract_info_from_filename(os.path.basename(model_path))
    args.dataset = dataset
    args.arch = model_name
    args.code_length = code_length
    model = load_model(args, args.code_length, erasing_model_path=model_path).to(args.device)
    # model.load_state_dict(torch.load(model_path))
    return model


def load_data_with_erasing(args, erasing_value):
    # Load dataset with specified erasing value
    args.erasing = erasing_value
    return load_data(args)


def evaluate_with_erasing(args, query_code, retrieval_code, query_targets, retrieval_targets, erasing_value):

    # Perform evaluation
    mAP = mean_average_precision(
        query_code.to(args.device),
        retrieval_code.to(args.device),
        query_targets.to(args.device),
        retrieval_targets.to(args.device),
        args.device,
        args.topk,
    )
    logger.info('[erasing:{}][map:{:.4f}]'.format(erasing_value, mAP))


def evaluate_model(args, model_weights_path):
    model = load_model_and_weights(model_weights_path, args)
    # Evaluate model
    model.eval()
    with torch.no_grad():
        args.soft=False
        args.cut=False
        _, query_dataloader, retrieval_dataloader = load_data_with_erasing(args, -1)
        # Determine code generation function based on architecture
        code_generation_function = generate_code_cut if 'cut' in args.arch else generate_code
        load_dict = torch.load(model_weights_path,
                               map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'})
        query_code = load_dict['qB']
        query_targets = load_dict['qL']
        retrieval_code = load_dict['rB']
        retrieval_targets = load_dict['rL']
        evaluate_with_erasing(args, query_code, retrieval_code, query_targets, retrieval_targets, 0)

        retrieval_code = code_generation_function(model, retrieval_dataloader, args.code_length, args.device)
        retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

        # flag Evaluate with different erasing values (0.4, 0.6, 0.8)
        for erasing_value in [0.2, 0.4, 0.6, 0.8]:
            # Load dataset with different erasing value
            _, query_dataloader, _ = load_data_with_erasing(args, erasing_value)
            query_code = code_generation_function(model, query_dataloader, args.code_length, args.device)
            evaluate_with_erasing(args, query_code, retrieval_code, query_targets, retrieval_targets, erasing_value)


def run_robustness(args, model_weights_paths):
    # GPU setting
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Robustness test
    for model_weights_path in model_weights_paths:
        model_info = os.path.splitext(os.path.basename(model_weights_path))[0]  # 提取模型名称
        # model_weights_path[12:-3]
        start_index = model_info.find('model_') + len('model_')
        end_index = model_info.find('_codelength')

        args.arch = model_info[start_index:end_index]

        if args.arch in ['resnet50', 'ViTHashing', 'ViTHashing_cut']:
            args.img_size = 224
        elif args.arch == 'googlenetv3':
            args.img_size = 299
        else:
            args.img_size = 240

        if 'cut' not in args.arch:
            args.layer_cut = -1


        pattern = r'(cifar-10|nus-wide-tc21|imagenet)'
        # 使用正则表达式进行匹配
        match = re.search(pattern, model_info)
        dataset_name = match.group(0)
        args.dataset = dataset_name
        if args.dataset == 'cifar-10':
            args.root = 'datasets/cifar-10'
            args.topk = -1
            args.num_class = 10
        elif args.dataset == 'nus-wide-tc21':
            args.root = 'datasets/NUS-WIDE'
            args.topk = 1000
            args.num_class = 21
        elif args.dataset == 'imagenet':
            args.root = 'datasets/Imagenet100'
            args.topk = 100
            args.num_class = 100
        model_log_path = create_model_log(model_info, args.dataset)  # 创建模型日志文件路径
        logger.add(model_log_path, rotation='500 MB', level='INFO')  # 添加日志记录器
        logger.info("Starting robustness test for model weights: {}\n"
                    "Dataset: {}".format(model_info, args.dataset)
                    )
        evaluate_model(args, model_weights_path)

if __name__ =="__main__":
    args = load_config()
    pth_resnet = [
        "1_Baseline/img/baseline_11_26_6_00_19_imagenet_model_resnet50_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.6293.pt",
                  "2_Others/img/others_6_16_5_23_32_imagenet_model_resnet50_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8316.pt",
                  "3_GT/img/crop_6_8_5_16_36_imagenet_model_resnet50_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8969.pt",
                  "4_GCUT/img/gcut_6_3_9_18_44_imagenet_model_resnet50_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8841.pt"
    ]

    pth_googlenet = [
        "1_Baseline/img/baseline_11_27_6_21_18_imagenet_model_googlenetv3_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.7418.pt",
        "2_Others/img/others_6_16_6_04_21_imagenet_model_googlenetv3_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.6846.pt",
        "3_GT/img/crop_6_8_10_21_36_imagenet_model_googlenetv3_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8355.pt",
        "4_GCUT/img/gcut_6_4_6_17_59_imagenet_model_googlenetv3_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.7790.pt"
        ]

    pth_msvit = [
    "1_Baseline/img/baseline_6_19_2_16_13_imagenet_model_crossvit_small_224_cut_debug_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8838.pt",
                  "2_Others/img/others_6_16_7_09_57_imagenet_model_crossvit_small_224_cut_debug_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8616.pt",
                  "3_GT/img/crop_7_3_1_18_26_imagenet_model_crossvit_small_224_cut_debug_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.9002.pt",
                  "4_GCUT/img/gct_6_3_11_10_48_imagenet_model_crossvit_small_224_cut_debug_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8931.pt"
                 ]

    pth_msvit_32 = [
        "1_Baseline/img/baseline_11_29_6_02_08_imagenet_model_crossvit_small_224_cut_debug_codelength_32_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8865.pt",
                  "2_Others/img/others_6_16_7_09_57_imagenet_model_crossvit_small_224_cut_debug_codelength_32_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8328.pt",
                  "3_GT/img/crop_6_8_3_03_41_imagenet_model_crossvit_small_224_cut_debug_codelength_32_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8927.pt",
                  "4_GCUT/img/gcut_6_23_3_18_39_imagenet_model_crossvit_small_224_cut_debug_codelength_32_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8930.pt"]

    pth_vit = ["1_Baseline/img/baseline_4_8_6_07_06_imagenet_model_ViTHashing_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.7335.pt",
               "2_Others/img/others_6_16_8_16_49_imagenet_model_ViTHashing_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.6598.pt",
               "3_GT/img/crop_6_8_4_10_38_imagenet_model_ViTHashing_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.7942.pt",
               "4_GCUT/img/gcut_6_15_4_15_44_imagenet_model_ViTHashing_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.7752.pt"]

    pth_naug = ["0_Non_aug/img/naug_6_8_1_15_45_imagenet_model_resnet50_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8186.pt",
               "0_Non_aug/img/naug_6_8_2_21_30_imagenet_model_googlenetv3_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.7249.pt",
               "0_Non_aug/img/naug_6_8_3_05_11_imagenet_model_crossvit_small_224_cut_debug_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.8850.pt",
               "0_Non_aug/img/naug_6_8_4_16_22_imagenet_model_ViTHashing_codelength_48_mu_0.01_nu_1_eta_0.01_topk_1000_map_0.7370.pt"]

    pth_files = pth_naug
    pth_files = [os.path.join("../../HDD/checkpoints/final_GG", file) for file in pth_files]
    run_robustness(args, pth_files)