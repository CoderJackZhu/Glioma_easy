import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import tqdm
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import gc
import nibabel as nib
import random
from tqdm import tqdm


# class ImageInfo:
#     def __init__(self, row):
#         self._data = row
#
#     @property
#     def path(self):
#         return self._data[0].split(' ')[0]
#
#     @property
#     def label(self):
#         return int(self._data[0].split(' ')[1])

def random_noise(arr_image, p):
    if random.random() < p:
        arr_image = noise(arr_image)
    return arr_image


def noise(arr_image):
    std = np.std(arr_image)
    noise = np.random.random(arr_image.shape)
    noise = 0.1 * std * 2 * (noise - 0.5)
    arr_image = arr_image + noise
    return arr_image


class RandomNoise(object):
    """
    把random_noise函数封装成类，方便transform.Compose使用
    """

    def __init__(self):
        pass

    def __call__(self, arr_img):
        return random_noise(arr_img, 0.5)


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


class My_resnet(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(My_resnet, self).__init__()
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
            nn.Linear(512, self.hidden_channels)
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
        self.model = My_resnet(input_channels, hidden_channels, num_classes)

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
        self.modalities = nn.ModuleList(
            [SingleModalityCNN(input_channels, hidden_channels, num_classes) for _ in range(num_modalities)])
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


class ClsDataset(Dataset):
    def __init__(self, list_file: str,
                 transform: list = None,
                 ):
        self.list_file = list_file
        self.transform = transform

        self._parser_input_data()

    def _parser_input_data(self):
        assert os.path.exists(self.list_file)

        lines = [x.strip().split('\t') for x in open(self.list_file, encoding='utf-8')]

        # 将列表中的每一项转换成路径和标签
        # self.imgs_list = [ImageInfo(line) for line in lines]
        self.imgs_list = lines
        print('Total images: ', len(self.imgs_list))

    def __getitem__(self, index):
        # 读取存放医学影像的文件夹
        img_dir_info = self.imgs_list[index]
        # 读取医学影像的路径
        img_path = img_dir_info[0].split(' ')[0]
        # 读取医学影像的标签
        img_label = int(img_dir_info[0].split(' ')[1])
        # 读取医学影像
        img_list = []
        path_dir = os.listdir(img_path)
        path_dir.sort()

        for file in path_dir:
            if file.endswith('.nii.gz'):
                img = nib.load(os.path.join(img_path, file)).get_fdata()
                img = np.array(img)
                if self.transform is not None:
                    for transform in self.transform:
                        img = transform(img)
                img_list.append(img)
        # img = np.array(img_list)
        # 把在通道维度上将四个模态的影像堆叠在一起，形成一个新的多通道影像，每个影像都素
        img = np.stack(img_list, axis=0)
        # # 将多个医学影像拼接在一起
        # img = np.concatenate(img, axis=2)
        # print(img.shape)
        return img, torch.as_tensor(img_label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs_list)


def train(train_loader, model, criterion, optimizer, scheduler, device, epoch, num_epoch):
    model.train()
    train_acc, train_loss = 0.0, 0.0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data = data.float()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(target.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()
    print(
        'Epoch:{}/{}\n Train： Average Loss: {:.6f},Accuracy:{:.2f}%'.format(epoch, num_epoch, train_loss / (batch_idx + 1),
                                                                         100.0 * train_acc / len(train_loader.dataset)))
    scheduler.step(train_acc)


def measure(test_loader, model, criterion, device, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    # writer = SummaryWriter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), test_loss, model.state_dict()


def imshow(epoch_list, loss_list, acc_list):
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.plot(epoch_list, loss_list, linestyle=':')
    plt.xlabel('epoch')
    plt.ylabel('Test loss')
    plt.subplot(122)
    plt.plot(epoch_list, acc_list, linestyle=':')
    plt.xlabel('epoch ')
    plt.ylabel('Test accuracy')
    plt.savefig('./result.png')
    plt.show()


if __name__ == '__main__':

    batch_size = 4
    num_classes = 4
    num_workers = 8
    train_list = '/media/spgou/DATA/ZYJ/Glioma_easy/dataset/train_patients.txt'
    val_list = '/media/spgou/DATA/ZYJ/Glioma_easy/dataset/test_patients.txt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalCNN(num_modalities=4, input_channels=4, hidden_channels=64, num_classes=4)
    print(model.state_dict().keys())
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)
    val_dataset = ClsDataset(
        list_file=val_list,
        transform=None
    )
    train_dataset = ClsDataset(
        list_file=train_list,
        transform=[RandomNoise()]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers, pin_memory=True,
                                               drop_last=True)
    num_epoch = 500

    max_acc = float('0')
    epoch_list, acc_list, loss_list = [], [], []
    for epoch in range(1, num_epoch + 1):
        train(train_loader, model, criterion, optimizer, scheduler, device, epoch, num_epoch)
        test_acc, test_ls, net_dict = measure(val_loader, model, criterion, device, epoch)
        epoch_list.append(epoch)
        loss_list.append(test_ls)
        acc_list.append(test_acc)
        if test_acc > max_acc:
            max_acc = test_acc
            # torch.save({'epoch_record': epoch, 'model': net_dict},f'./模型/动物分类模型_{max_acc}%.pth')
            torch.save(model, f'./models_{max_acc}%.pth')
        imshow(epoch_list, loss_list, acc_list)
