import os
import h5py
import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.transform import *


class ImageInfo:
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0].split(',')[0]

    @property
    def label(self):
        return self._data[0].split(',')[1]


def split_train_test(glioma_dir='/media/spgou/DATA/ZYJ/Dataset/captk_before_data',
                     annotate_file='PathologicalData_DropNull_manualCorrected_analyzed_anonymized.xlsx'):
    """
    处理数据并得到划分好的训练集列表和测试集txt文件，文件的每行是路径加标签
    :return:
    """
    # 读取表格中的数据
    annotate_file = pd.read_excel(annotate_file, header=0)
    patients = annotate_file['PatientID_anonymized'].values
    # labels = annotate_file['WHO_grade'].values
    labels = annotate_file['WHO_grade'].values
    labels = np.array(labels, dtype=str)
    # 删除标签为nan的数据，并删除对应的病人ID
    patients = np.delete(patients, np.where(labels == 'nan'))
    labels = np.delete(labels, np.where(labels == 'nan'))
    labels = np.array(labels, dtype=float)

    # # 把标签更改为0，1，2，3
    # labels[np.where(labels == 1)] = 0
    # labels[np.where(labels == 2)] = 1
    # labels[np.where(labels == 3)] = 2
    # labels[np.where(labels == 4)] = 3

    # 把前三类标签合并为一类，最后一类，为二分类问题
    labels[np.where(labels == 1)] = 0
    labels[np.where(labels == 2)] = 0
    labels[np.where(labels == 3)] = 0
    labels[np.where(labels == 4)] = 1

    # 读取图片的路径
    imgs_path = []
    img_label = []
    dirs = os.listdir(glioma_dir)
    for dir in dirs:
        patient_id = '_'.join(dir.split('_')[:2])
        if patient_id in patients:
            imgs_path.append(os.path.join(glioma_dir, dir))
            img_label.append(labels[np.where(patients == patient_id)[0][0]])

    # 输出低级别和高级别的数量
    print('Low grade: ', len(np.where(np.array(img_label) == 0)[0]))
    print('High grade: ', len(np.where(np.array(img_label) == 1)[0]))

    # 将数据分为训练集和测试集
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs_path, img_label, test_size=0.2,
                                                                        random_state=3407, stratify=img_label)
    print('Train Data: ', len(train_imgs))
    print('Test Data: ', len(test_imgs))

    # 将训练集和测试集的数据分别写入txt文件
    with open('train_patients.txt', 'w') as f:
        for i in range(len(train_imgs)):
            f.write(train_imgs[i] + ' ' + str(int(train_labels[i])) + '\n')
    with open('test_patients.txt', 'w') as f:
        for i in range(len(test_imgs)):
            f.write(test_imgs[i] + ' ' + str(int(test_labels[i])) + '\n')


def split_train_test_1p19q(glioma_dir='/media/spgou/DATA/ZYJ/Dataset/captk_before_data',
                           annotate_file='PathologicalData_DropNull_manualCorrected_analyzed_anonymized.xlsx'):
    """
    处理数据并得到划分好的训练集列表和测试集txt文件，文件的每行是路径加标签
    :return:
    """
    # 读取表格中的数据
    annotate_file = pd.read_excel(annotate_file, header=0)
    patients = annotate_file['PatientID_anonymized'].values
    # labels = annotate_file['WHO_grade'].values
    labels = annotate_file['analyze_mutation_1p19q'].values
    labels = np.array(labels, dtype=str)
    # 删除标签为nan的数据，并删除对应的病人ID
    patients = np.delete(patients, np.where(labels == 'nan'))
    labels = np.delete(labels, np.where(labels == 'nan'))
    labels = np.array(labels, dtype=float)

    # # 把标签更改为0，1，2，3
    # labels[np.where(labels == 1)] = 0
    # labels[np.where(labels == 2)] = 1
    # labels[np.where(labels == 3)] = 2
    # labels[np.where(labels == 4)] = 3

    # 把前三类标签合并为一类，最后一类，为二分类问题
    # labels[np.where(labels == 1)] = 0
    # labels[np.where(labels == 2)] = 0
    # labels[np.where(labels == 3)] = 0
    # labels[np.where(labels == 4)] = 1

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

    # 输出低级别和高级别的数量
    print('Low grade: ', len(np.where(np.array(img_label) == 0)[0]))
    print('High grade: ', len(np.where(np.array(img_label) == 1)[0]))

    # 将数据分为训练集和测试集
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs_path, img_label, test_size=0.2,
                                                                        random_state=3407, stratify=img_label)
    print('Train Data: ', len(train_imgs))
    print('Test Data: ', len(test_imgs))

    # 将训练集和测试集的数据分别写入txt文件
    with open('train_patients.txt', 'w') as f:
        for i in range(len(train_imgs)):
            f.write(train_imgs[i] + ' ' + str(int(train_labels[i])) + '\n')
    with open('test_patients.txt', 'w') as f:
        for i in range(len(test_imgs)):
            f.write(test_imgs[i] + ' ' + str(int(test_labels[i])) + '\n')


def xiangya_split_train_val_test_IDH(glioma_dir='/media/spgou/DATA/ZYJ/Dataset/captk_before_data',
                            annotate_file='PathologicalData_anonymized_20231027.xlsx'):
        """
        处理数据并得到划分好的训练集列表和测试集txt文件，文件的每行是路径加标签
        :return:
        """
        # 读取表格中的数据
        annotate_file = pd.read_excel(annotate_file, header=0)
        patients = annotate_file['PatientID_anonymized'].values
        # labels = annotate_file['WHO_grade'].values
        labels = annotate_file['analyze_mutation_IDH'].values
        labels = np.array(labels, dtype=str)
        # 删除标签为nan的数据，并删除对应的病人ID
        patients = np.delete(patients, np.where(labels == 'nan'))
        labels = np.delete(labels, np.where(labels == 'nan'))
        labels = np.array(labels, dtype=float)


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

        # 输出变异和非变异的数量
        print('IDH mutation: ', len(np.where(np.array(img_label) == 1)[0]))
        print('IDH wild-type: ', len(np.where(np.array(img_label) == 0)[0]))

        # 将数据分为训练集、验证集和测试集
        train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(imgs_path, img_label, test_size=0.2,
                                                                            random_state=3407, stratify=img_label)
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(train_val_imgs, train_val_labels, test_size=0.2,
                                                                            random_state=3407, stratify=train_val_labels)
        print('Train Data: ', len(train_imgs))
        print('Val Data: ', len(val_imgs))
        print('Test Data: ', len(test_imgs))

        # 将训练集和测试集的数据分别写入txt文件
        with open('train_patients.txt', 'w') as f:
            for i in range(len(train_imgs)):
                f.write(train_imgs[i] + ' ' + str(int(train_labels[i])) + '\n')
        with open('val_patients.txt', 'w') as f:
            for i in range(len(val_imgs)):
                f.write(val_imgs[i] + ' ' + str(int(val_labels[i])) + '\n')
        with open('test_patients.txt', 'w') as f:
            for i in range(len(test_imgs)):
                f.write(test_imgs[i] + ' ' + str(int(test_labels[i])) + '\n')



def TCGA_train_test_split(image_dir='G:\Dataset\TCGA-TCIA-ArrangedData/TCIA/Images'):
    annotate_dir = '/media/spgou/DATA/ZYJ/Dataset/TCGA-TCIA-ArrangedData/ArrangedGeneData/TCGA_subtypes_IDH.xlsx'
    # 读取表格中的数据
    annotate_file = pd.read_excel(annotate_dir, header=0)
    patients = annotate_file['patient_id'].values
    is_GBM = annotate_file['is_GBM'].values
    IDH = annotate_file['is_IDH_mutant'].values
    is_1p19q = annotate_file['is_1p19q_codeleted'].values

    # 对1p19q和IDH进行筛选
    is_1p19q[np.where(IDH == 0)] = 0
    # 全设为-1
    gene_subtype = np.ones(patients.shape[0], dtype=int) * -1
    for i in range(patients.shape[0]):
        if IDH[i] == 0:
            gene_subtype[i] = 0
        elif IDH[i] == 1 and is_1p19q[i] == 0:
            gene_subtype[i] = 1
        elif IDH[i] == 1 and is_1p19q[i] == 1:
            gene_subtype[i] = 2

    # 212 patients have the tumor grade, IDH mutation, and 1p/19q codeletion info.
    #
    # 1）is_GBM:
    # - 1：glioblastoma(GBM);
    # - 0：Low Grade Glioma(LGG);
    #
    # 2) is_IDH_mutant:
    # - 1: IDH mutant;
    # - 0: IDH wild-type;
    #
    # 3) is_1p19q_codeleted:
    # - 1: 1p/19q co-deleted;
    # - 0: 1p/19q intact;

    # # 删除标签为nan的数据，并删除对应的病人ID
    # patients = np.delete(patients, np.where(is_GBM == 'nan'))
    # IDH = np.delete(IDH, np.where(is_GBM == 'nan'))
    # is_1p19q = np.delete(is_1p19q, np.where(is_GBM == 'nan'))
    # is_GBM = np.delete(is_GBM, np.where(is_GBM == 'nan'))
    #
    # # 删除标签为unknown的数据，并删除对应的病人ID
    # patients = np.delete(patients, np.where(is_GBM == 'unknown'))
    # IDH = np.delete(IDH, np.where(is_GBM == 'unknown'))
    # is_1p19q = np.delete(is_1p19q, np.where(is_GBM == 'unknown'))
    # is_GBM = np.delete(is_GBM, np.where(is_GBM == 'unknown'))

    # print('GBM: ', len(np.where(is_GBM == 1)[0]))
    # print('LGG: ', len(np.where(is_GBM == 0)[0]))

    # 读取图片的路径
    imgs_path = []
    img_label = []
    dirs = os.listdir(image_dir)
    for dir in dirs:
        if dir in patients:
            imgs_path.append(os.path.join(image_dir, dir))
            label_1 = IDH[np.where(patients == dir)[0][0]]
            # label_2 = is_1p19q[np.where(patients == dir)[0][0]]
            # label_3 = is_GBM[np.where(patients == dir)[0][0]]
            # img_label.append([label_1, label_2, label_3])
            # img_label.append(gene_subtype[np.where(patients == dir)[0][0]])
            img_label.append(label_1)
    # 将数据分为训练集和测试集
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs_path, img_label, test_size=0.2,
                                                                        random_state=3407, stratify=img_label)
    print('Train Data: ', len(train_imgs))
    print('Test Data: ', len(test_imgs))

    # 将训练集和测试集的数据分别写入txt文件
    # with open('tcia_ucsf_train_patients.txt', 'w') as f:
    #     for i in range(len(train_imgs)):
    #         f.write(train_imgs[i] + ' ' + str(train_labels[i][0]) + ' ' + str(train_labels[i][1]) + ' ' + str(
    #             train_labels[i][2]) + '\n')
    # with open('tcia_ucsf_test_patients.txt', 'w') as f:
    #     for i in range(len(test_imgs)):
    #         f.write(test_imgs[i] + ' ' + str(test_labels[i][0]) + ' ' + str(test_labels[i][1]) + ' ' + str(
    #             test_labels[i][2]) + '\n')
    with open('tcia_ucsf_train_patients.txt', 'w') as f:
        for i in range(len(train_imgs)):
            f.write(train_imgs[i] + ',' + str(train_labels[i]) + '\n')
    with open('tcia_ucsf_test_patients.txt', 'w') as f:
        for i in range(len(test_imgs)):
            f.write(test_imgs[i] + ',' + str(test_labels[i]) + '\n')


def ucsf_train_test_split(img_dir='/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images'):
    annotate_file = pd.read_csv('/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-metadata_v2.csv', header=0)
    patients = annotate_file['ID'].values
    is_1p19q = annotate_file['1p/19q'].values
    # 加一列编码后的标签
    num_1p19q = np.ones(patients.shape[0], dtype=int) * -1
    # 对其进行编码, co-deleted relative co-deleted,: 1, intact: 0
    # is_1p19q[np.where(is_1p19q != 'co-deleted' and is_1p19q != 'relative co-deleted')] = 1

    num_1p19q[np.where(is_1p19q == 'intact')] = 0
    num_1p19q[np.where(is_1p19q == 'relative co-deletion')] = 1
    num_1p19q[np.where(is_1p19q == 'co-deletion')] = 1
    num_1p19q[np.where(is_1p19q == 'Co-deletion')] = 1

    is_IDH = annotate_file['IDH'].values
    num_IDH = np.ones(patients.shape[0], dtype=int) * -1
    # 对其进行编码, wildtype: 0, mutated(NOS): 1
    num_IDH[np.where(is_IDH == 'wildtype')] = 0
    # is_IDH[np.where(is_IDH == 'mutated (NOS)')] = 1
    # IDH非空的情况下，除了wildtype，其他都是mutated
    num_IDH[np.where(is_IDH != 'wildtype')] = 1

    # 对1p19q和IDH进行筛选
    num_1p19q[np.where(num_IDH == 0)] = 0
    # 全设为-1
    gene_subtype = np.ones(patients.shape[0], dtype=int) * -1
    for i in range(patients.shape[0]):
        if num_IDH[i] == 0:
            gene_subtype[i] = 0
        elif num_IDH[i] == 1 and num_1p19q[i] == 0:
            gene_subtype[i] = 1
        elif num_IDH[i] == 1 and num_1p19q[i] == 1:
            gene_subtype[i] = 2

    # 删除标签为genotype为-1的数据，并删除对应的病人ID
    patients = np.delete(patients, np.where(gene_subtype == -1))
    num_1p19q = np.delete(num_1p19q, np.where(gene_subtype == -1))
    num_IDH = np.delete(num_IDH, np.where(gene_subtype == -1))
    gene_subtype = np.delete(gene_subtype, np.where(gene_subtype == -1))

    # # 新加两列
    # annotate_file['num_1p19q'] = num_1p19q
    # annotate_file['num_IDH'] = num_IDH
    # # 新加一列geng_subtype
    # annotate_file['gene_subtype'] = gene_subtype
    # annotate_file.to_csv('/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-metadata_add_gene.csv', index=False)
    # who_grade = annotate_file['WHO CNS Grade'].values
    # # 把2,3,4变换为0,1,2
    # who_grade[np.where(who_grade == 2)] = 0
    # who_grade[np.where(who_grade == 3)] = 1
    # who_grade[np.where(who_grade == 4)] = 2

    # 删除标签为nan的数据，并删除对应的病人ID
    # patients = np.delete(patients, np.where(who_grade == 'nan'))
    # is_1p19q = np.delete(is_1p19q, np.where(who_grade == 'nan'))
    # is_IDH = np.delete(is_IDH, np.where(who_grade == 'nan'))
    # who_grade = np.delete(who_grade, np.where(who_grade == 'nan'))

    # 删除标签为unknown的数据，并删除对应的病人ID
    # patients = np.delete(patients, np.where(who_grade == 'unknown'))
    # is_1p19q = np.delete(is_1p19q, np.where(who_grade == 'unknown'))
    # is_IDH = np.delete(is_IDH, np.where(who_grade == 'unknown'))
    # who_grade = np.delete(who_grade, np.where(who_grade == 'unknown'))

    # 读取图片的路径
    imgs_path = []
    img_label = []
    dirs = os.listdir(img_dir)
    for dir in dirs:
        patient_id = dir.split('_')[0]
        patient_id = '-'.join(patient_id.split('-')[:2]) + '-' + patient_id.split('-')[-1][1:]
        if patient_id in patients:
            imgs_path.append(os.path.join(img_dir, dir))
            # label1 = who_grade[np.where(patients == patient_id)[0][0]]
            # label1 = gene_subtype[np.where(patients == patient_id)[0][0]]
            label1 = num_IDH[np.where(patients == patient_id)[0][0]]
            img_label.append(label1)

    # 将数据分为训练集和测试集
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs_path, img_label, test_size=0.2,
                                                                        random_state=3407, stratify=img_label)
    print('Train Data: ', len(train_imgs))
    print('Test Data: ', len(test_imgs))

    # 将训练集和测试集的数据分别写入txt文件
    with open('tcia_ucsf_train_patients.txt', 'a') as f:
        for i in range(len(train_imgs)):
            f.write(train_imgs[i] + ',' + str(int(train_labels[i])) + '\n')
    with open('tcia_ucsf_test_patients.txt', 'a') as f:
        for i in range(len(test_imgs)):
            f.write(test_imgs[i] + ',' + str(int(test_labels[i])) + '\n')


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
                 transform: list = None,
                 ):
        self.list_file = list_file
        self.transform = transform
        self.h5py_path = h5py_path
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
        img_data_path = os.path.join(self.h5py_path, patient + '.h5')
        # img = h5py.File(img_data_path, 'r')
        with h5py.File(img_data_path, 'r') as f:
            img = f['0'][:]

        # 读取医学影像的标签
        img_label = self.imgs_list[index].label
        if len(img_label) == 1:
            img_label = int(img_label[0])
        else:
            img_label = np.array(img_label, dtype=float)
        if self.transform is not None:
            for transform in self.transform:
                img = transform(img)
        # # 将多个医学影像拼接在一起
        # img = np.concatenate(img, axis=2)
        # print(img.shape)
        return img, torch.as_tensor(img_label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs_list)


if __name__ == '__main__':
    # split_train_test(
    #     glioma_dir='/media/spgou/DATA/ZYJ/Dataset/captk_before_data_zscore_normalizedImages_have_seg_ROI_images_expand_rm_blank')
    # # train_dataset = ClsDataset(list_file='train_patients.txt', transform=[Resize((128, 128, 128))])
    # split_train_test(glioma_dir="/media/spgou/DATA/ZYJ/Dataset/zscore_normalizedImages_ROI_images_expand",
    #                  annotate_file="/media/spgou/DATA/ZYJ/Glioma_easy/dataset/PathologicalData_anonymized_20231027.xlsx")
    # split_train_test(glioma_dir="/media/spgou/DATA/ZYJ/Dataset/captk_before_data_zscore_normalizedImages_have_seg_ROI_images_expand_rm_blank")
    # test_dataset = ClsDataset(list_file='test_patients.txt', transform=[Resize((128, 128, 128)),
    # # #                                                                     # RandomAugmentation((16, 16, 16), (0.8, 1.2),(0.8, 1.2)),
    # ])
    # train_dataset = ClsDataset(list_file='train_patients.txt', transform=[Resize((128, 128, 128)),
    #                                                                       RandomAugmentation((16, 16, 16), (0.8, 1.2),(0.8, 1.2)),
    #                                                                       ])
    TCGA_train_test_split('/media/spgou/DATA/ZYJ/Dataset/TCGA-TCIA-ArrangedData_ROI_images_expand')
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

    # TCGA_train_test_split('/media/spgou/DATA/ZYJ/Dataset/TCGA-TCIA-ArrangedData/TCIA/Images')
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
    ucsf_train_test_split('/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images')

    # save_as_h5py('/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images',
    #              '/media/spgou/FAST/UCSF_TCIA_ROI_images_h5py')
    # save_as_h5py('/media/spgou/DATA/ZYJ/Dataset/TCGA-TCIA-ArrangedData_ROI_images_expand',
    #              '/media/spgou/FAST/UCSF_TCIA_ROI_images_h5py')

    # train_dataset = ClsDatasetH5py(list_file='ucsf_train_patients.txt',
    #                                h5py_path='/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images_h5py',
    #                                transform=[Resize((128, 128, 128))])
    # for k, v in train_dataset:
    #     print('Training:', k.shape)
