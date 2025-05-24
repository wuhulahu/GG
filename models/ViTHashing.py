import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights
from models.modules import generate_att_mask_convs, FeatureEncoder
import torch.nn.functional as F

class ViT(nn.Module):
    def __init__(self, code_length, weight_remain, sigma, limit, cut=False, use_custom_activation=False, ratio_att=True):
        super(ViT, self).__init__()
        self.weight_remain = weight_remain  # 保留的比例
        # self.features_map_encoder = FeatureEncoder('crossvit_small_224')
        self.sigma = sigma
        self.limit = limit
        self.cut = cut
        self.use_custom_activation = use_custom_activation
        self.ratio_att = ratio_att
        # vit_b_16
        # vit_weights = ViT_B_16_Weights.DEFAULT
        # self.vit_b = vit_b_16(weights=vit_weights)
        # vit_b_32
        vit_weights = ViT_B_32_Weights.DEFAULT
        self.vit_b = vit_b_32(weights=vit_weights)

        self.hash_layer = nn.Sequential(
            nn.Linear(1000, code_length, bias=True),
            nn.Tanh()
        )
        self._init_weights()
    def _init_weights(self):
        for layer in self.hash_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='tanh')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def get_att_mask(self, x):
        img_shape = x.shape
        # 生成feature_map

        self.eval()
        # x = self.features_map_encoder(x)
        x = self.vit_b._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit_b.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit_b.encoder(x)

        # 生成att_token
        att_score = x.detach()[:, 0:1] @ x.detach()[:, 1:].permute(0, 2, 1)

        att_score = att_score.reshape(img_shape[0], 7, 7)
        expanded_mask, adjust_ratio = generate_att_mask_convs(att_score, img_shape, self.weight_remain, self.sigma, self.limit, use_thresholds=False, vit=True,
                                                              use_custom_activation=self.use_custom_activation, ratio_att=self.ratio_att)
        return expanded_mask, adjust_ratio

    def forward(self, x):
        if self.cut and self.training:
            with torch.no_grad():
                self.eval()
                t = x.detach()
                t, adjust_ratio = self.get_att_mask(t)
                x = x * t
                self.train()
        else:
            adjust_ratio = None

        # vit_b_16.forward
        x = self.vit_b._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit_b.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit_b.encoder(x)
        x = self.vit_b.heads(x[:, 0])

        out = self.hash_layer(x)
        if self.training:
            return out, adjust_ratio
        else:
            return out

def vit_load(arch, code_length, weight_remain, sigma, limit, cut, use_custom_activation=False, ratio_att=True, erasing_model_path=None):

    model = ViT(code_length, weight_remain, sigma, limit, cut, use_custom_activation, ratio_att)


    if erasing_model_path is not None:
        model.load_state_dict(torch.load(erasing_model_path, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'})['model'])
    return model