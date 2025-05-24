import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torch.nn.functional as F
from utils.tools import num_sample
from models.modules import generate_att_mask_convs, FeatureEncoder


def load_model(code_length, weight_remain=0.75, cut=False, sigma=3, limit=0.5, use_custom_activation=False, ratio_att=True, erasing_model_path=None):

    googlenet_layer = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model = GoogleNetV3(googlenet_layer, code_length, weight_remain, cut, sigma, limit, use_custom_activation, ratio_att)
    if erasing_model_path is not None:
        model.load_state_dict(torch.load(erasing_model_path, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'})['model'],strict=False)

    return model


class GoogleNetV3(nn.Module):
    def __init__(self, features, code_length, weight_remain, cut, sigma, limit, use_custom_activation, ratio_att):
        super(GoogleNetV3, self).__init__()
        self.weight_remain = weight_remain
        self.cut = cut
        self.sigma = sigma
        self.limit = limit
        self.use_custom_activation = use_custom_activation
        self.ratio_att = ratio_att
        self.num_sample = num_sample
        self.features = features
        # self.features_map_encoder = FeatureEncoder('GoogleNetV3')
        self.hash_layer = nn.Sequential(
            nn.Linear(1000, code_length),
            nn.Tanh(),
        )
        self.init_custom_layers()

    def init_custom_layers(self):
        for module in self.hash_layer:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Use appropriate initialization for your model
                    init.xavier_normal_(m.weight)
                    init.constant_(m.bias, 0.0)

    def get_att_mask(self, x):
        img_shape = x.shape
        # feature_map
        # x = self.features_map_encoder(x)

        # 可训练
        x = self.features.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.features.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.features.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.features.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.features.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.features.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.features.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.features.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.features.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.features.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.features.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.features.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.features.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.features.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.features.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.features.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.features.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.features.Mixed_7c(x)

        # 生成att_mask和调整比例
        expanded_mask, adjust_ratio = generate_att_mask_convs(x, img_shape, self.weight_remain, self.sigma, self.limit, use_custom_activation=self.use_custom_activation, ratio_att=self.ratio_att)
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
            adjust_ratio=None

        if self.training:
            x, _ = self.features(x)
        else:
            x = self.features(x)
        x = self.hash_layer(x)


        if self.training:
            return x, adjust_ratio
        else:
            return x