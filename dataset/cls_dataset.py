import nibabel as nib
from sklearn.model_selection import train_test_split
from dataset.transform import *
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
from skimage import measure
import pandas as pd
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
import math
import h5py


def cc2weight(cc, w_min: float = 1., w_max: float = 2e5):
    weight = torch.zeros_like(cc, dtype=torch.float32)
    cc_items = torch.unique(cc)
    K = len(cc_items) - 1
    N = torch.prod(torch.tensor(cc.shape))
    for i in cc_items:
        weight[cc == i] = N / ((K + 1) * torch.sum(cc == i))
    return torch.clip(weight, w_min, w_max)


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, imagec):
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)
        return image


class Random_Flip(object):
    def __call__(self, image):

        if random.random() < 0.5:
            image = np.flip(image, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)

        return image


from scipy.ndimage import zoom


class Random_Crop(object):
    def __call__(self, image):
        H = random.randint(0, 128-64)
        W = random.randint(0, 128-64)
        D = random.randint(0, 128-64)

        image = image[H: H + 64, W: W + 64, D: D + 64, ...]
        # image = image[61: 61 + 128, 61: 61 + 128, 11: 11 + 128, ...]

        return image


class Random_intencity_shift(object):
    def __call__(self, image, factor=0.1):
        scale_factor = np.random.uniform(1.0 - factor, 1.0 + factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image * scale_factor + shift_factor

        return image


class Random_rotate(object):
    def __call__(self, image):
        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)

        return image


class Pad(object):
    def __call__(self, image):
        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        return image
    # (240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float()

        return image


class Augmentation(object):
    """Augmentation for the training data.

   :array: A numpy array of size [c, x, y, z]
   :returns: augmented image and the corresponding mask

   """

    def __call__(self, array):
        array = array.transpose(3, 0, 1, 2)
        # 无mask，所以mask为全图
        mask = np.ones(array.shape[1:], dtype=bool)
        # normalize image to range [0, 1], then apply this transform
        patch_size = np.asarray(array.shape)[1:]
        augmented = noise_transforms.augment_gaussian_noise(
            array, noise_variance=(0, .015))

        # need to become [bs, c, x, y, z] before augment_spatial
        augmented = augmented[None, ...]
        mask = mask[None, None, ...]
        r_range = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.

        augmented, mask = spatial_transforms.augment_spatial(
            augmented, seg=mask, patch_size=patch_size,
            do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
            do_scale=True, scale=(.9, 1.1),
            border_mode_data='constant', border_cval_data=cval,
            order_data=3,
            p_el_per_sample=0.5,
            p_scale_per_sample=.5,
            p_rot_per_sample=.5,
            random_crop=False
        )
        # mask = mask[0][0]
        image = augmented[0]
        image = image.transpose(1, 2, 3, 0)
        return image


def transform(sample):
    trans = transforms.Compose([
        # Pad(),
        # Random_rotate(),  # time-consuming
        # Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        Augmentation(),
        ToTensor()
    ])
    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        # Pad(),
        # MaxMinNormalization(),
        # Random_Crop(),
        ToTensor()
    ])

    return trans(sample)


def transform_test(sample):
    trans = transforms.Compose([
        # Pad(),
        # MaxMinNormalization(),
        # Random_Crop(),
        ToTensor()
    ])

    return trans(sample)


class ImageInfo:
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0].split(',')[0]

    @property
    def label(self):
        return self._data[0].split(',')[1]


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

        # 将列表中的每一项转换成ImageInfo类
        self.imgs_list = [ImageInfo(line) for line in lines]
        # print('Load total images: ', len(self.imgs_list))
        print(f'Load {os.path.basename(self.list_file).split(".")[0]} total images: {len(self.imgs_list)}')

    def __getitem__(self, index):
        # 读取存放医学影像的文件夹
        img_dir_info = self.imgs_list[index]
        # 读取医学影像的路径
        img_path = img_dir_info.path
        # 读取医学影像的标签
        img_label = img_dir_info.label
        if len(img_label) == 1:
            img_label = int(img_label[0])
        else:
            img_label = np.array(img_label, dtype=float)
        # 读取医学影像
        img_list = []
        path_dir = os.listdir(img_path)
        path_dir.sort()
        # i = 0
        # 只读取一个影像
        # path_dir = path_dir[1:2]
        # print(path_dir)
        for file in path_dir:
            if file.endswith('.nii.gz'):
                img = nib.load(os.path.join(img_path, file)).get_fdata()
                img = np.array(img)
                # 把最后一维的通道数放在第一维
                img = np.transpose(img, (2, 0, 1))
                # print(img.shape)
                # print(img.shape)

                # # 把最后一维的通道数放在第一维
                # img = np.transpose(img, (3, 0, 1, 2))
                # print(img.shape)
                img_list.append(img)
                # i += 1
                # if i == 1:
                #     break
        # img = np.array(img_list)
        # 把在通道维度上将四个模态的影像堆叠在一起，形成一个新的多通道影像，每个影像都素
        img = np.stack(img_list, axis=0)
        if self.transform is not None:
            for transform in self.transform:
                img = transform(img)
        # # 将多个医学影像拼接在一起
        # img = np.concatenate(img, axis=2)
        # print(img.shape)
        return img, torch.as_tensor(img_label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs_list)


def save_as_h5py(img_path, h5py_path):
    """
    把每个文件夹中的影像读取出来，并将其转换成h5py文件，每个文件夹对应一个h5py文件
    Args:
        h5py_path:
        img_path:

    Returns:

    """
    if not os.path.exists(h5py_path):
        os.makedirs(h5py_path)
    patients = os.listdir(img_path)
    patients.sort()
    for patient in tqdm.tqdm(patients):
        # 读取文件夹中的影像
        img_list = []
        path_dir = os.listdir(os.path.join(img_path, patient))

        path_dir.sort()
        for file in path_dir:
            if file.endswith('.nii.gz'):
                img = nib.load(os.path.join(img_path, patient, file)).get_fdata()
                img = np.array(img, dtype=np.float32)
                # 把最后一维的通道数放在第一维
                img = np.transpose(img, (2, 0, 1))
                img_list.append(img)

        # 把在通道维度上将四个模态的影像堆叠在一起，形成一个新的多通道影像，每个影像都素
        img = np.stack(img_list, axis=0)
        # 保存为h5py文件
        with h5py.File(os.path.join(h5py_path, patient + '.h5'), 'w') as f:
            f.create_dataset('0', data=img)


#
# def save_as_npy(img_path, npy_path):
#     """
#         把每个文件夹中的影像读取出来，并将其转换成成npy文件，每个文件夹对应一个npy文件
#     """
#     if not os.path.exists(npy_path):
#         os.makedirs(npy_path)
#     patients = os.listdir(img_path)
#     patients.sort()
#     for patient in tqdm.tqdm(patients):
#         # 读取文件夹中的影像
#         img_list = []
#         path_dir = os.listdir(os.path.join(img_path, patient))
#         path_dir.sort()
#         for file in path_dir:
#             if file.endswith('.nii.gz'):
#                 img = nib.load(os.path.join(img_path, patient, file)).get_fdata()
#                 print(img.dtype)
#                 img = np.array(img)
#                 # 查看精度
#
#                 # 把最后一维的通道数放在第一维
#                 img = np.transpose(img, (2, 0, 1))
#                 img_list.append(img)
#
#         # 把在通道维度上将四个模态的影像堆叠在一起，形成一个新的多通道影像，每个影像都素
#         img = np.stack(img_list, axis=0)
#         # 对数据进行压缩，减少存储空间
#         img = np.array(img, dtype=np.float32)
#
#         # 保存为npy文件
#         np.save(os.path.join(npy_path, patient + '.npy'), img)
#
#
#
#
class ClsDatasetH5py(Dataset):
    def __init__(self, list_file: str,
                 h5py_path: str = "./h5py",
                 mode: str = 'train',
                 ):
        self.list_file = list_file
        self.transform = transform
        self.h5py_path = h5py_path
        self.mode = mode
        if h5py_path is None:
            raise ValueError('h5py_path must be set when use_h5py is True')
        self._parser_input_data()

    def _parser_input_data(self):
        """
        读取存放医学影像的文件夹，并将其转换成h5py文件，每个文件夹对应一个h5py文件

        Returns:

        """
        assert os.path.exists(self.list_file)

        lines = [x.strip().split('\t') for x in open(self.list_file, encoding='utf-8')]
        # 将列表中的每一项转换成ImageInfo类
        self.imgs_list = [ImageInfo(line) for line in lines]
        print('Total images: ', len(self.imgs_list))

    def __getitem__(self, index):
        # 读取存放h5py格式的医学影像的文件夹
        img_path = self.imgs_list[index]
        # 读取医学影像的路径
        img_path = img_path.path
        patient = os.path.basename(img_path)
        img_data_path = os.path.join(self.h5py_path, patient + '_nifti.h5')
        # img = h5py.File(img_data_path, 'r')
        with h5py.File(img_data_path, 'r') as f:
            img = f['0'][:]

        # 读取医学影像的标签
        img_label = self.imgs_list[index].label
        if len(img_label) == 1:
            img_label = int(img_label[0])
        else:
            img_label = np.array(img_label, dtype=float)
        # if self.transform is not None:
        #     for transform in self.transform:
        if self.mode == 'train':
            img = transform(img)
        elif self.mode == 'val':
            img = transform_valid(img)
        elif self.mode == 'test':
            img = transform_test(img)
        # # 将多个医学影像拼接在一起
        # img = np.concatenate(img, axis=2)
        # print(img.shape)
        return img, torch.as_tensor(img_label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        return [int(x.label) for x in self.imgs_list]


# if __name__ == '__main__':

# test_dataset = ClsDataset(list_file='test_patients.txt', transform=[Resize((128, 128, 128)),
# # #                                                                     # RandomAugmentation((16, 16, 16), (0.8, 1.2),(0.8, 1.2)),
# ])
# train_dataset = ClsDataset(list_file='train_patients.txt', transform=[Resize((128, 128, 128)),
#                                                                       RandomAugmentation((16, 16, 16), (0.8, 1.2),(0.8, 1.2)),
#                                                                       ])

# # test_dataset = ClsDataset(list_file='tcia_test_patients.txt', transform=[Resize((128, 128, 128))])
# train_dataset = ClsDataset(list_file='tcia_train_patients.txt', transform=[Resize((128, 128, 128)),
#                                                                              RandomAugmentation((16, 16, 16),
#                                                                                                 (0.8, 1.2),
#                                                                                                 (0.8, 1.2)),
#                                                                                 ])
# #
# for k, v in test_dataset:
#     print('Training:', k.shape)
# save_test_dir = './test_out'
# save_train_dir = './train_tcga_transform_out'
# j = 0
# for k, v in test_dataset:
#     # print('Testing:', k.shape, v)
#     # 把k按照第一个通道的维度拆分成四个模态，并保存为nii文件
#     # 每个k建立一个文件夹
#     save_file_dir = os.path.join(save_test_dir, str(j))
#     os.makedirs(save_file_dir, exist_ok=True)
#     for i in range(k.shape[0]):
#         img = nib.Nifti1Image(k[i, :, :, :], np.eye(4))
#         nib.save(img, os.path.join(save_file_dir, str(i) + '.nii.gz'))
#     j += 1

# j = 0
# for k, v in train_dataset:
#     print('Training:', k.shape, v)
#     # 把k按照第一个通道的维度拆分成四个模态，并保存为nii文件
#     # 每个k建立一个文件夹
#     save_file_dir = os.path.join(save_train_dir, str(j))
#     os.makedirs(save_file_dir, exist_ok=True)
#     for i in range(k.shape[0]):
#         img = nib.Nifti1Image(k[i, :, :, :], np.eye(4))
#         nib.save(img, os.path.join(save_file_dir, str(i) + '.nii.gz'))
#     j += 1

# # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
# # 获取一个batch的数据并可视化
# # for i, (data, label) in enumerate(test_loader):
# #     print(data.shape)
# #     print(label.shape)
# #     print(label)
# #     print(data)
# #     break
# one_batch = next(iter(test_loader))
# print(one_batch[0].shape)
# # 把数据转换为numpy数组
# one_batch = one_batch[0].numpy()
# print(one_batch.shape)
# # 将其保存为nii文件
# img = nib.Nifti1Image(one_batch[0, 0, :, :, :], np.eye(4))
# nib.save(img, 'test.nii.gz')
# print(one_batch.shape)
# img = nib.load('F:\\Code\\Medical\\Glioma_easy\\test_data_out\\Gliomas_00012_20190906_T1.nii.gz').get_fdata()
# print(img.shape)


# train_dataset = ClsDataset(list_file='tcia_train_patients.txt', transform=[Resize((128, 128, 128),
#                                                                                   orig_shape=(155, 240, 240))])
#
# for k, v in train_dataset:
#     print('Training:', k.shape, v)
# save_as_h5py('/media/spgou/DATA/ZYJ/Dataset/captk_before_data_zscore_normalizedImages_have_seg_ROI_images_expand_rm_blank',
#              '/media/spgou/DATA/ZYJ/Dataset/captk_before_data_zscore_normalizedImages_have_seg_ROI_images_expand_rm_blank_h5py')
# save_as_npy('G:\\Dataset\\Xiangya_data\\captk_before_data_zscore_normalizedImages',
#             'G:\\Dataset\\Xiangya_data\\captk_before_data_zscore_normalizedImages_npy')
# save_as_h5py('/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images',
#              '/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images_h5py')


# save_as_h5py('/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images',
#              '/media/spgou/FAST/UCSF_TCIA_ROI_images_h5py')
# save_as_h5py('/media/spgou/DATA/ZYJ/Dataset/TCGA-TCIA-ArrangedData_ROI_images_expand',
#              '/media/spgou/FAST/UCSF_TCIA_ROI_images_h5py')
if __name__ == '__main__':
    train_dataset = ClsDatasetH5py(list_file='/media/spgou/FAST/ZYJ/Glioma_MTTU/data/ucsf_train_patients.txt',
                                   h5py_path='/media/spgou/FAST/UCSF/UCSF-PDGM-v3-20230111_ROI_images_h5py',
                                   mode='train')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)

    # for k, v in train_dataset:
    #     print('Training:', k.shape)
    # 获取一个batch的数据并可视化
    # for i, (data, label) in enumerate(test_loader):
    #     print(data.shape)
    #     print(label.shape)
    #     print(label)
    #     print(data)
    #     break
    one_batch = next(iter(train_loader))
    print(one_batch[0].shape)
    # 把数据转换为numpy数组
    one_batch = one_batch[0].numpy()
    print(one_batch.shape)
    # 将其保存为nii文件
    img = nib.Nifti1Image(one_batch[0, 0, :, :, :], np.eye(4))
    nib.save(img, 'test.nii.gz')

