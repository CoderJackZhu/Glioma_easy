
from sklearn.model_selection import train_test_split

import os

import numpy as np

import pandas as pd



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

if __name__ == '__main__':
    # split_train_test(
    #     glioma_dir='/media/spgou/DATA/ZYJ/Dataset/captk_before_data_zscore_normalizedImages_have_seg_ROI_images_expand_rm_blank')
    # # train_dataset = ClsDataset(list_file='train_patients.txt', transform=[Resize((128, 128, 128))])
    # split_train_test(glioma_dir="/media/spgou/DATA/ZYJ/Dataset/zscore_normalizedImages_ROI_images_expand",
    #                  annotate_file="/media/spgou/DATA/ZYJ/Glioma_easy/dataset/PathologicalData_anonymized_20231027.xlsx")
    # split_train_test(glioma_dir="/media/spgou/DATA/ZYJ/Dataset/captk_before_data_zscore_normalizedImages_have_seg_ROI_images_expand_rm_blank")
    TCGA_train_test_split('/media/spgou/DATA/ZYJ/Dataset/TCGA-TCIA-ArrangedData_ROI_images_expand')
    # TCGA_train_test_split('/media/spgou/DATA/ZYJ/Dataset/TCGA-TCIA-ArrangedData/TCIA/Images')
    ucsf_train_test_split('/media/spgou/DATA/ZYJ/Dataset/5.UCSF-PDGM/UCSF-PDGM-v3-20230111_ROI_images')