from logging import raiseExceptions
import torch
import torch.nn as nn
import torchvision

ModelWeights = {
    'mobilenet_v2': 'MobileNet_V2_Weights.IMAGENET1K_V1',
    'resnet18': 'ResNet18_Weights.IMAGENET1K_V1',
    'resnet50': 'ResNet50_Weights.IMAGENET1K_V1',
    'resnet101': 'ResNet101_Weights.IMAGENET1K_V1',
    'swin_s': 'Swin_S_Weights.IMAGENET1K_V1',
    'swin_b': 'Swin_B_Weights.IMAGENET1K_V1',
    'vit_b_16': 'ViT_B_16_Weights.IMAGENET1K_V1',
    'vit_b_32': 'ViT_B_32_Weights.IMAGENET1K_V1',
    'vit_l_16': 'ViT_L_16_Weights.IMAGENET1K_V1',
    'vit_l_32': 'ViT_L_32_Weights.IMAGENET1K_V1'
}


class ClsModel(nn.Module):
    def __init__(self, model_name, num_classes, is_pretrained=False):
        super(ClsModel, self).__init__()
        self.model_name = model_name
        self.num_class = num_classes
        self.is_pretrained = is_pretrained

        if self.model_name not in ModelWeights:
            raise ValueError('Please confirm the name of model')

        if self.is_pretrained:
            self.base_model = getattr(torchvision.models, self.model_name)(weights=ModelWeights[self.model_name])
        else:
            self.base_model = getattr(torchvision.models, self.model_name)()

        if hasattr(self.base_model, 'classifier'):
            self.base_model.last_layer_name = 'classifier'
            feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'fc'):
            self.base_model.last_layer_name = 'fc'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.fc = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'head'):
            self.base_model.last_layer_name = 'head'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.head = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'heads'):
            self.base_model.last_layer_name = 'heads'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.heads = nn.Linear(feature_dim, self.num_class)
        else:
            raise ValueError('Please confirm the name of last')

    #         self.new_fc = nn.Linear(feature_dim, self.num_class)

    def forward(self, x):
        x = self.base_model(x)
        #         x = self.new_fc(x)
        return x


# 定义单个模态的卷积神经网络
class SingleModalityCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SingleModalityCNN, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc = nn.Linear(64 * 4 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4 * 4)
        x = self.fc(x)
        return x


# 定义多模态模型，特征级融合
class MultiModalCNN(nn.Module):
    def __init__(self, num_modalities, input_channels, num_classes):
        super(MultiModalCNN, self).__init__()
        self.modalities = nn.ModuleList([SingleModalityCNN(input_channels, num_classes) for _ in range(num_modalities)])
        self.fc = nn.Linear(num_modalities * num_classes, num_classes)

    def forward(self, inputs):
        modal_outputs = [modality(inputs[:, i, :, :, :]) for i, modality in enumerate(self.modalities)]
        fused_output = torch.cat(modal_outputs, dim=1)
        fused_output = self.fc(fused_output)
        return fused_output


if __name__ == '__main__':
    model_name = 'resnet50'
    num_classes = 4
    is_pretrained = False

    clsmodel = ClsModel(model_name, num_classes, is_pretrained)
    print(clsmodel)
