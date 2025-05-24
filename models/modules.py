import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.tools import num_sample
from models import model_loader
from torch.hub import load_state_dict_from_url

def my_sigmoid(x, scale=2.0, shift=0.0):
    sigmoid_half = torch.sigmoid(scale * torch.tensor(0.5) + shift)
    offset = sigmoid_half - 0.5
    return torch.sigmoid(scale * x + shift) - offset


def custom_activation(x):
    # return torch.where(x > 0.5, x, torch.sigmoid(x)-0.5)
    return torch.where(x > 0.5, x, my_sigmoid(x))

def generate_att_mask_convs(x, img_shape, weight_remain, sigma, limit, use_thresholds=True, vit=False, use_custom_activation=False, ratio_att=True):
    '''
    生成面向convs的att_mask,利用类激活图实现
    remain： 保留的图像比例
    sigma： 和vit的sigma效果一样
    use_thresholds: 是否使用阈值来判断保留的目标
    vit： 是否是为vit模型生成mask
    '''
    b, c, ini_h, ini_w = img_shape
    # sum
    # x_sum = x.sum(dim=1, keepdim=True)
    # x_sum = x_sum.view(128, 7, 7)
    # softmax_t_sum = F.softmax(x_sum.view(128, -1), dim=-1).view(128, 7, 7)

    # mean
    if not vit:
        x = x.mean(dim=1, keepdim=True).squeeze(1)
        # 将特征图扩大到（12，12）

        x = F.interpolate(x.unsqueeze(1), size=(12, 12), mode='nearest').squeeze(1)

    _, h, w = x.shape
    x = x.view(b, -1)
    x = F.softmax(x, dim=1)

    if use_thresholds and not vit:
        thresholds = x.mean(dim=1, keepdim=True)
        # 使用 torch.where() 找到大于阈值的索引
        idx_range = [torch.where(x[i] > thresholds[i])[0] for i in range(len(x))]
    else:
        sorted_indices = torch.argsort(x, dim=1, descending=True)

        # 计算每行的保留索引数量，即每行的长度的 75%
        num_to_keep = int(x.shape[1] * weight_remain)

        # 截取每行的索引，保留最大的 75% 的索引
        idx_range = sorted_indices[:, :num_to_keep]

    mask_idx = []
    adjust_ratio = []
    for i, idx_one_sample in enumerate(idx_range):
        num_remain_tokens = idx_one_sample.size(-1)
        num_mask = num_sample(int(num_remain_tokens * sigma), num_remain_tokens * limit)
        rand_indices = torch.randperm(num_remain_tokens)
        # 取前 num_mask 个随机索引
        sampled_indices_idx_one_sample = rand_indices[:num_mask]
        mask_idx.append(idx_one_sample[sampled_indices_idx_one_sample])
        if ratio_att:
            sum_cut_att_score = x[i][idx_one_sample[sampled_indices_idx_one_sample]].sum()
            adjust_ratio_att = 1 - sum_cut_att_score
            if use_custom_activation:
                adjust_ratio_att = custom_activation(adjust_ratio_att)

            adjust_ratio.append(adjust_ratio_att)
        else:   # ratio_num_remain_tokens
            adjust_ratio.append(1 - num_mask / num_remain_tokens)

    mask = torch.ones_like(x, requires_grad=False)
    for i, sampled in enumerate(mask_idx):
        mask[i][sampled] = 0.

    mask = mask.view(b, h, w)
    expanded_mask = F.interpolate(mask.unsqueeze(1), size=(ini_h, ini_w), mode='nearest')
    return expanded_mask, torch.tensor(adjust_ratio).to(x.device)


class FeatureEncoder(nn.Module):
    """
    外挂特征编码器，用于提取feature_map
    """
    def __init__(self, arch):
        self.arch = arch
        super(FeatureEncoder, self).__init__()
        if arch == 'GoogleNetV3':
            self.features_map_encoder = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        elif arch == 'resnet50':
            self.features_map_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif arch =='crossvit_small_224':
            # self.features_map_encoder = model_loader.load_model('crossvit_small_224', code_length=16)
            from timm.models import create_model
            self.features_map_encoder = create_model(
                'crossvit_small_224',
                pretrained=True,
                num_classes=1000,
                drop_rate=0.0,
                drop_path_rate=0.1,
                code_length=16,
                drop_block_rate=None,
            )
        elif arch == "ViTHashing":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            vit_weights = ViT_B_16_Weights.DEFAULT
            self.features_map_encoder = vit_b_16(weights=vit_weights)
        else:
            raise ValueError("Invalid encoder name")
        for param in self.features_map_encoder.parameters():
            param.requires_grad = False

    def GoogleNetV3_forward(self, x):
        self.features_map_encoder.eval()
        x = self.features_map_encoder.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.features_map_encoder.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.features_map_encoder.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.features_map_encoder.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.features_map_encoder.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.features_map_encoder.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.features_map_encoder.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.features_map_encoder.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.features_map_encoder.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.features_map_encoder.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.features_map_encoder.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.features_map_encoder.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.features_map_encoder.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.features_map_encoder.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.features_map_encoder.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.features_map_encoder.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.features_map_encoder.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.features_map_encoder.Mixed_7c(x)
        return x

    def resnet50_forward(self, x):
        self.features_map_encoder.eval()
        x = self.features_map_encoder.conv1(x)
        x = self.features_map_encoder.bn1(x)
        x = self.features_map_encoder.relu(x)
        x = self.features_map_encoder.maxpool(x)
        x = self.features_map_encoder.layer1(x)
        x = self.features_map_encoder.layer2(x)
        x = self.features_map_encoder.layer3(x)
        x = self.features_map_encoder.layer4(x)
        return x

    def crossvit_small_224_forward(self, x):
        self.features_map_encoder.eval()
        B, C, H, W = x.shape
        xs = []
        for i in range(self.features_map_encoder.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(self.features_map_encoder.img_size[i], self.features_map_encoder.img_size[i]), mode='bicubic') if H != \
                                                                                                                  self.features_map_encoder.img_size[
                                                                                                                      i] else x
            tmp = self.features_map_encoder.patch_embed[i](x_)
            cls_tokens = self.features_map_encoder.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.features_map_encoder.pos_embed[i]
            tmp = self.features_map_encoder.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.features_map_encoder.blocks:
            xs = blk(xs)
        return xs

    def vit_forward(self, x):
        x = self.features_map_encoder._process_input(x)
        n = x.shape[0]
        batch_class_token = self.features_map_encoder.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.features_map_encoder.encoder(x)
        return x

    def forward(self, x):
        self.eval()
        if self.arch == 'GoogleNetV3':
            x = self.GoogleNetV3_forward(x)
        elif self.arch == 'resnet50':
            x = self.resnet50_forward(x)
        elif self.arch == 'crossvit_small_224':
            x = self.crossvit_small_224_forward(x)
        elif self.arch == 'ViTHashing':
            x = self.vit_forward(x)
        return x