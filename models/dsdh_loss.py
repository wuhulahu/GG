import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class DSDHLoss(nn.Module):
    def __init__(self, eta, cl_loss):
        super(DSDHLoss, self).__init__()
        self.eta = eta
        self.cl_loss = cl_loss

    def forward(self, U_batch, U, S, B, W, targets, soft_weight=None):
        # theta：[num_train, batch_size] ,相似矩阵，这个batch和所有train_data344
        theta = U.t() @ U_batch / 2

        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)
        if soft_weight != None:
            S = S * soft_weight.view(1,-1)
            targets = targets*soft_weight.view(-1, 1)

        metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()
        # metric_loss = soft_target(U, S)
        quantization_loss = (B - U_batch).pow(2).mean()

        if self.cl_loss:
            cl_loss = (targets.t() - W.t() @ B).pow(2).mean() + W.pow(2).mean()

            loss = cl_loss + metric_loss + self.eta * quantization_loss
        else:
            loss = metric_loss + self.eta * quantization_loss
        return loss


def get_center_point(inputs, mask, idx_pos_max, flag='remove_pos_max'):
    centers = []
    for i in range(inputs.shape[0]):
        this_mask = mask[i].t()
        this_pos_max = idx_pos_max[i]
        # 取出与i标签相同的batch_data,并删掉pos_max
        this_inputs = inputs[this_mask]
        if flag == 'remove_pos_max':
            this_inputs = torch.cat((this_inputs[:this_pos_max-1], this_inputs[this_pos_max:]))
        this_center = this_inputs.mean(dim=0)
        centers.append(this_center.unsqueeze(dim=0))

    return torch.cat(centers)


def get_pair_mask(S):
    # s_labels是这个batch的、t_labels是所有的ground_true
    batch_size = S.shape[0]
    num_data = S.shape[1]
    # sim_origin = s_labels.mm(t_labels.t())
    sim_origin = S
    sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., num_data) + 2
    ph = ph.repeat(batch_size, 1).reshape(batch_size, num_data)
    th = torch.log2(ph).cuda() # 这里可能会有问题，改为'.to(S.device)'
    IDCG = ((2 ** ideal_list - 1) / th).sum(axis=1)
    DCG = ((2 ** sim_origin - 1)/th).sum(axis=1)
    ma3 = torch.max(DCG)
    mi3 = torch.min(DCG)
    NDCG = torch.div(IDCG , DCG)
    weight = NDCG
    #mask = sim
    return weight

class TripletLoss(nn.Module):   # with query_centers
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, batch_size=128, view_num=3, p=2, dataset_name="cifar10", code_length=16, query_center=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.similarity = torch.cat([torch.arange(batch_size) for i in range(view_num)], dim=0)
        self.tri = nn.TripletMarginLoss(margin=0.3)
        # self.tri = nn.TripletMarginLoss(margin=0.3, reduction='none')
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        if self.dataset_name == 'cifar-10':
            self.num_class = 10
            self.K = 25
        elif self.dataset_name == 'nus-wide-tc21':
            self.num_class = 21
            self.K = 25
        elif self.dataset_name == 'imagenet':
            self.num_class = 100
            self.K = 8
        elif self.dataset_name == 'coco':
            self.num_class = 80
            self.K = 8

    def forward(self, inputs, similarity=None, soft_weight=None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        if similarity == None:
            similarity = self.similarity
        n = inputs.size(0)
        mask = similarity != 0

        centers = []
        for i in range(inputs.shape[0]):
            this_mask = mask[i].t()
            # 取出与i标签相同的batch_data,并删掉pos_max
            this_inputs = inputs[this_mask]
            this_center = this_inputs.mean(dim=0)
            centers.append(this_center.unsqueeze(dim=0))
        centers = torch.cat(centers)

        dist = []
        for i in range(n):
            # dist.append(inputs[i] - inputs[targets != targets[i]])
            dist.append(centers[i] - inputs)
        dist = torch.stack(dist)
        dist = torch.linalg.norm(dist, ord=self.p, dim=2)


        # 取出每个input样本的hard_pos, hard_neg
        pos_max, neg_min = [], []
        no_neg_idx = []
        no_pos_idx = []
        for i in range(n):
            idx_d_min_neg = dist[i][~mask[i]]

            if idx_d_min_neg.numel() == 0:
                no_neg_idx.append(i)
                continue
            if mask[i].sum() == 0:
                no_pos_idx.append(i)
                continue
            idx_d_min_neg = idx_d_min_neg.argmin()
            neg_min.append(inputs[~mask[i].t()][idx_d_min_neg].unsqueeze(0))
        neg_min = torch.cat(neg_min, dim=0)

        # 去掉没有neg样本的数据，主要发生在multilabel
        if len(no_neg_idx) != 0 or len(no_pos_idx) != 0:
            # inputs = torch.cat([row.unsqueeze(0) for i, row in enumerate(inputs) if i not in no_neg_idx], dim=0)
            # mask = torch.cat([row.unsqueeze(0) for i, row in enumerate(mask) if i not in no_neg_idx], dim=0)
            #
            inputs = torch.stack([inputs[i] for i in range(len(inputs)) if i not in no_neg_idx and i not in no_pos_idx])
            # centers也需要跳过没有正样例和负样例的数据
            centers = torch.stack([centers[i] for i in range(len(centers)) if i not in no_neg_idx and i not in no_pos_idx])
            # mask = torch.stack([mask[i] for i in range(len(mask)) if i not in no_neg_idx])

            keep_idx = [i for i in range(mask.size(0)) if i not in no_neg_idx]
            # 使用这些索引进行切片操作，删除指定的行和列
            mask = mask[keep_idx][:, keep_idx]

        tri_loss = self.tri(centers, inputs, neg_min)

        return tri_loss.mean()
