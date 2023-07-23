import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split


class ImageInfo:
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0].split(' ')[0]

    @property
    def label(self):
        return int(self._data[0].split(' ')[1])


def split_train_test(glioma_dir='/media/spgou/DATA/ZYJ/Dataset/captk_before_data',
                     annotate_file='PathologicalData_DropNull_manualCorrected_analyzed_anonymized.xlsx'):
    """
    处理数据并得到划分好的训练集列表和测试集txt文件，文件的每行是路径加标签
    :return:
    """
    # 读取表格中的数据
    annotate_file = pd.read_excel(annotate_file, header=0)
    patients = annotate_file['PatientID_anonymized'].values
    labels = annotate_file['WHO_grade'].values
    labels = np.array(labels, dtype=str)
    # 删除标签为nan的数据，并删除对应的病人ID
    patients = np.delete(patients, np.where(labels == 'nan'))
    labels = np.delete(labels, np.where(labels == 'nan'))
    labels = np.array(labels, dtype=float)

    # 把标签更改为0，1，2，3
    labels[np.where(labels == 1)] = 0
    labels[np.where(labels == 2)] = 1
    labels[np.where(labels == 3)] = 2
    labels[np.where(labels == 4)] = 3

    # 将数据分为训练集和测试集
    # train_patients, test_patients, train_labels, test_labels = train_test_split(patients, labels, test_size=0.2,
    #                                                                             random_state=0)
    # 读取图片的路径
    imgs_path = []
    img_label = []
    dirs = os.listdir(glioma_dir)
    for dir in dirs:
        patient_id = '_'.join(dir.split('_')[:2])
        if patient_id in patients:
            imgs_path.append(os.path.join(glioma_dir, dir))
            img_label.append(labels[np.where(patients == patient_id)[0][0]])

    # 将数据分为训练集和测试集
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs_path, img_label, test_size=0.2,
                                                                        random_state=0)

    # 将训练集和测试集的数据分别写入txt文件
    with open('train_patients.txt', 'w') as f:
        for i in range(len(train_imgs)):
            f.write(train_imgs[i] + ' ' + str(int(train_labels[i])) + '\n')
    with open('test_patients.txt', 'w') as f:
        for i in range(len(test_imgs)):
            f.write(test_imgs[i] + ' ' + str(int(test_labels[i])) + '\n')


class ClsDataset(Dataset):
    def __init__(self, list_file, transform=None):
        self.list_file = list_file
        self.transform = transform
        self._parser_input_data()

    def _parser_input_data(self):
        assert os.path.exists(self.list_file)

        lines = [x.strip().split('\t') for x in open(self.list_file, encoding='utf-8')]

        # 将列表中的每一项转换成ImageInfo类
        self.imgs_list = [ImageInfo(line) for line in lines]
        print('Total images: ', len(self.imgs_list))

    def __getitem__(self, index):
        # 读取存放医学影像的文件夹
        img_dir_info = self.imgs_list[index]
        # 读取医学影像的路径
        img_path = img_dir_info.path
        # 读取医学影像的标签
        img_label = img_dir_info.label
        # 读取医学影像
        img_list = []
        path_dir = os.listdir(img_path)
        path_dir.sort()

        for file in path_dir:
            if file.endswith('.nii.gz'):
                img = nib.load(os.path.join(img_path, file)).get_fdata()
                img = np.array(img)
                img_list.append(img)
        # img = np.array(img_list)
        # 把在通道维度上将四个模态的影像堆叠在一起，形成一个新的多通道影像，每个影像都素
        img = np.stack(img_list, axis=0)
        # # 将多个医学影像拼接在一起
        # img = np.concatenate(img, axis=2)
        # print(img.shape)

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.as_tensor(img_label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs_list)


if __name__ == '__main__':
    split_train_test(glioma_dir='G:\\Dataset\\captk_before_data')
    # train_dataset = ClsDataset(list_file='train_patients.txt', transform=None)
    # test_dataset = ClsDataset(list_file='test_patients.txt', transform=None)
    #
    # for k, v in train_dataset:
    #     print('Training:', k, v)
    #
    # for k, v in test_dataset:
    #     print('Testing:', k, v)
