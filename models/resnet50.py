import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
from utils.tools import num_sample
from models.modules import generate_att_mask_convs, FeatureEncoder


def load_model(code_length, weight_remain=0.75, cut=False, sigma=3, limit=0.5, use_custom_activation=False, ratio_att=True, erasing_model_path=None):
    resnet_layer = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = resnet50(resnet_layer, code_length, weight_remain, cut, sigma, limit, use_custom_activation, ratio_att)
    if erasing_model_path is not None:
        model.load_state_dict(torch.load(erasing_model_path, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'})['model'], strict=False)

    return model


class resnet50(nn.Module):

    def __init__(self, features, code_length, weight_remain, cut, sigma, limit, use_custom_activation, ratio_att):
        super(resnet50, self).__init__()
        self.weight_remain = weight_remain
        self.cut = cut
        self.sigma = sigma
        self.limit = limit
        self.use_custom_activation = use_custom_activation
        self.ratio_att = ratio_att
        self.num_sample = num_sample
        self.features = features
        # self.features_map_encoder = FeatureEncoder('resnet50')
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
        # # 生成feature_map
        # x = self.features_map_encoder(x)
        # 可训练
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)
        x = self.features.layer1(x)
        x = self.features.layer2(x)
        x = self.features.layer3(x)
        x = self.features.layer4(x)

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

        x = self.features(x)
        x = self.hash_layer(x)

        if self.training:
            return x, adjust_ratio
        else:
            return x