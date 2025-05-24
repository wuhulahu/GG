import torch
from thop import profile
from thop import clever_format
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import numpy as np
import re
import subprocess
import time
import requests
import datetime


def count_parameters(model, input_data):

    input_tensor = torch.randn(1, *input_data.shape[1:])
    macs, params = profile(model, inputs=(input_tensor,))

    # Convert the parameter count to millions
    params_in_million = params / 1e6

    print(f"Model Parameter Count: {params_in_million:.2f}M")


def num_sample(sigma=50, limit=180, n=100):
    # draw an integer from gaussian within +/- limit
    for d in range(n):
        x = torch.randn((1)) * sigma
        if abs(x) <= limit:
            return int(abs(x))
    # return int(0)
    return int(limit)


def showimg(idxs, x, adjust=None):
    '''
    from utils.tools import showimg
    showimg([8], x, adjust_ratio)
    '''
    for idx in idxs:
        if adjust is not None:
            print(adjust[idx])

        img_np = transforms.ToPILImage()(x[idx])
        plt.imshow(img_np)
        plt.axis('off')  # 关闭坐标轴
        plt.show()


def visualize_attention_batch(x, t):
    attention_matrix = torch.softmax(t, dim=1)
    x_weighted = x * attention_matrix
    x_weighted_normalized = torch.clamp(x_weighted, 0, 1)

    return x_weighted_normalized


def extract_info_from_filename(filename):
    pattern = r'(\d+_\d+_\d+_\d+:\d+)_(.+)_model_(.+)_codelength_(\d+)_.*map_([\d.]+)\.pt'
    match = re.match(pattern, filename)
    if match:
        datetime = match.group(1)
        dataset = match.group(2)
        model = match.group(3)
        code_length = match.group(4)
        map_value = match.group(5)
        return datetime, dataset, model, code_length, map_value
    else:
        return None


def get_gpu_memory_usage(memory_threshold):
    # 获取GPU总显存量
    total_gpu_memory = int(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader']).decode().strip())
    # 获取GPU已使用的显存量
    used_gpu_memory_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    # 解析输出并计算总显存使用量
    used_gpu_memory = sum(int(line.strip()) for line in used_gpu_memory_output.decode().split('\n') if line.strip())
    # 计算可用的GPU显存
    available_gpu_memory = total_gpu_memory - used_gpu_memory
    return available_gpu_memory > memory_threshold


warning_flag = False
def check_idle_gpu_memory(memory_threshold, waiting_time):
    # print("Idle GPU memory is insufficient. Waiting...")
    # while not get_gpu_memory_usage(memory_threshold):
    #     time.sleep(waiting_time)
    global warning_flag
    if not get_gpu_memory_usage(memory_threshold) and not warning_flag:
        print("Idle GPU memory is insufficient. Waiting...")
        warning_flag = True

    while not get_gpu_memory_usage(memory_threshold):
        time.sleep(waiting_time)

def push_notification(mark, error_type=None):

    """
    Bark in ios app store
    """
    if 'debug' in mark:
        return
    # 获取当前时间
    current_time = datetime.datetime.now()

    # 格式化当前时间
    formatted_time = current_time.strftime("%m-%d %H:%M")

    if error_type is not None:
        ret_iphone = requests.get()
        ret_mac = requests.get()
    else:
        ret_iphone = requests.get()

        ret_mac = requests.get()

    if ret_iphone.status_code == 200:
        print("推送完成！")
    else:
        print("无法推送！")


