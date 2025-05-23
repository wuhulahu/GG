# Gain from Give Up: Intuitive Data Augmentation Framework for Image Retrieval

**GG** is a general and plug-and-play data augmentation framework designed to improve deep hashing and real-valued retrieval tasks by explicitly modeling and mitigating semantic mismatches introduced during training. Unlike conventional augmentation strategies that assume label preservation, GG identifies and selectively discards semantically inconsistent regions to provide more informative gradients during learning.

> 🚩 *Discard to gain*: GG introduces a semantic retention score to evaluate augmented content and integrate it into pairwise losses for more robust retrieval optimization.

---


## REQUIREMENTS
We use python to build our code, you need to install those package to run

- pytorch 1.12.1
- sklearn
- tqdm
- pillow

## DATASETS
1. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3
3. [Imagenet100](https://pan.baidu.com/s/1Vihhd2hJ4q0FOiltPA-8_Q) Password: ynwf
4. [COCO](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) (include 2017 train,val and annotations)

### Processing dataset
To accelerate training and reduce data loading overhead, we recommend preprocessing raw datasets into .npy format files. This approach significantly improves I/O efficiency during both training and evaluation, especially when dealing with large-scale datasets like NUS-WIDE and COCO.
After all mat file generated, the dir of `dataset` will like this:

## Running Experiments
We provide two scripts for running experiments:
### 🔹 `run.py`: Run a single experiment
This script allows you to launch a single experiment by specifying the dataset, model architecture, and augmentation options through command-line arguments.

**Example:**

```bash
python run.py --dataset coco --arch resnet50 --soft True
```
You can view all available options by checking the argument parser in `run.py`.

### 🔹 `run_list_2.py`: Run multiple experiments in batch

This script supports **automated batch execution** across combinations of datasets, models, and augmentations. It is useful for ablation studies and large-scale evaluations.

**Usage:**

```bash
python run_list_2.py
```

## Acknowledgments

🧬 This project is developed based on the following open-source repositories.We sincerely thank the original author for making the code publicly available:

- [DSDH](https://github.com/Tree-Shu-Zhao/DSDH_PyTorch)
- [DCHMT](https://github.com/kalenforn/DCHMT)
