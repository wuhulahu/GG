# Gain from Give Up: Intuitive Data Augmentation Framework for Image Retrieval *(under review)*

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

## Note on $G_{cut}$ Efficiency

The $G_{cut}$ module uses attention-based saliency to perform targeted cropping on key semantic regions. By design, this
introduces **an additional forward pass** to compute attention maps, which may increase computational overhead.

### Approximate Optimization (Proposed, Not Yet Implemented)

In response to reviewer feedback, we propose a lightweight approximation to mitigate this overhead. The core idea is to
**reuse the attention maps generated in the previous epoch** instead of recomputing them each time:

- In the middle to later stages of training, model attention tends to stabilize.
- Stored attention maps from the previous epoch can still provide reliable semantic guidance.
- Attention maps are lightweight and inexpensive to store and apply.

This approximation eliminates the additional forward pass while preserving most of the semantic benefit of $G_{cut}$.

| Variant                          | Extra Forward Pass | Attention Cost                                                          | Total Complexity (per sample)                                                   |
|----------------------------------|--------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| $G_t$                            | ❌                  | None                                                                    | $\mathcal{O}(1)$                                                                |
| **Original $G_{cut}$**           | ✅                  | $\mathcal{O}(c \cdot h \cdot w)$ (CNN) / $\mathcal{O}(h \cdot w)$ (ViT) | $\mathcal{O}(n + c \cdot h \cdot w)$ (CNN) / $\mathcal{O}(n + h \cdot w)$ (ViT) |
| **Approx. $G_{cut}$** | ❌                  | Same as above                                                           | $\mathcal{O}(c \cdot h \cdot w)$ / $\mathcal{O}(h \cdot w)$                     |

#### Notation:

- $c$: number of channels in the feature map
- $h \times w$: spatial resolution of the feature map used for attention computation
- $n$: complexity of a single forward pass through the backbone network (feature extraction)
- All values are theoretical per-sample costs

This approximation substantially reduces computational burden while preserving semantic guidance, as the cost of feature
extraction ($\mathcal{O}(n)$) dominates the attention computation. It is therefore well-suited for large-scale or
time-constrained training pipelines. This strategy was inspired by a reviewer suggestion and is documented here as part of our ongoing optimization efforts.


## Acknowledgments

🧬 This project is developed based on the following open-source repositories.We sincerely thank the original author for making the code publicly available:

- [DSDH](https://github.com/Tree-Shu-Zhao/DSDH_PyTorch)
- [DCHMT](https://github.com/kalenforn/DCHMT)
