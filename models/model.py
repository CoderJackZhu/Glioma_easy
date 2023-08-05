from logging import raiseExceptions
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

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

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class my_resnet(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(my_resnet, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(240, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            resnet_block(32, 32, 2, True),
            resnet_block(32, 64, 2),
            resnet_block(64, 128, 2),
            resnet_block(128, 256, 2),
            resnet_block(256, 512, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )

    def forward(self, x):
        x = self.net(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define single modality 3D CNN
class SingleModalityCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(SingleModalityCNN, self).__init__()
        # self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool3d(2)
        # self.fc = nn.Linear(64 * 30 * 30 * 19, num_classes)
        # self.model = torchvision.models.resnet34(pretrained=True)
        # 使用3d resnet提取特征，让网络的输入通道数为1
        # self.model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        # self.model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model = my_resnet(input_channels, hidden_channels, num_classes)



    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 64 * 30 * 30 * 19)
        # x = self.fc(x)

        x = self.model(x)
        # print(x.shape)
        return x

# Define multi-modal model with feature-level fusion
class MultiModalCNN(nn.Module):
    def __init__(self, num_modalities, input_channels, hidden_channels, num_classes):
        super(MultiModalCNN, self).__init__()
        self.modalities = nn.ModuleList([SingleModalityCNN(input_channels, hidden_channels, num_classes) for _ in range(num_modalities)])
        # 把得到的一维特征向量进行通道拼接并融合
        self.fusion = nn.Sequential(
            nn.Linear(num_modalities * hidden_channels, num_modalities * hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(num_modalities * hidden_channels, num_classes)

    def forward(self, inputs):
        modal_outputs = [modality(inputs[:, i, :, :, :]) for i, modality in enumerate(self.modalities)]
        fused_output = torch.cat(modal_outputs, dim=1)
        fused_output = self.fusion(fused_output)
        fused_output = self.fc(fused_output)
        return fused_output


if __name__ == '__main__':
    # model_name = 'resnet50'
    # num_classes = 4
    # is_pretrained = False
    #
    # clsmodel = ClsModel(model_name, num_classes, is_pretrained)
    # print(clsmodel)
    num_modalities = 4
    input_channels = 4
    num_classes = 4
    hidden_channels = 64
    device = torch.device('cuda:0')
    model = MultiModalCNN(num_modalities, input_channels, hidden_channels, num_classes).to(device)
    batch_size = 1
    inputs = torch.randn(batch_size, 240, 240, 155, num_modalities).to(device)
    # 通道变换
    # inputs = inputs.permute(0, 4, 1, 2, 3)
    # one_model = SingleModalityCNN(input_channels, num_classes)
    # print(inputs[:, 1, :, :, :].shape)
    # output = model(inputs[:, 1, :, :, :])
    # one_model=one_model.to(device)
    outputs = model(inputs)
    # print(one_model(inputs[:, 1, :, :, :]).shape)
    print(outputs.shape)
    print(outputs.cpu().detach().numpy())