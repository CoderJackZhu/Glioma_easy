# -*- coding:utf-8 -*-
# @PROJECT_NAME :Glioma_easy
# @FileName     :ROI_region.py
# @Time         :2023/9/7 19:20
# @Author       :Jack Zhu

import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


#
#
# def get_max_3d_ROI(source_dir, mask_file, target_dir):
#     """
#     对于源目录下的所有3D图像，根据掩码文件ROI掩码获取最大的方框，并将其保存到目标目录下。
#     """
#     source_dir = Path(source_dir)
#     target_dir = Path(target_dir)
#     target_dir.mkdir(parents=True, exist_ok=True)
#
#     # 读取掩码文件
#     mask = sitk.ReadImage(mask_file)
#
#     # 获取包围掩码ROI的最大3D方框
#     max_ROI = get_ROI__box(mask)
#
#     # 遍历源目录下的所有3D图像
#     for image_file in tqdm.tqdm(source_dir.glob('*.nii.gz')):
#         # 读取3D图像
#         image = sitk.ReadImage(str(image_file))
#
#         # 裁剪3D图像
#         cropped_image = crop_image(image, max_ROI)
#
#         # 保存裁剪后的3D图像
#         target_file = target_dir / image_file.name
#         sitk.WriteImage(cropped_image, str(target_file))
#
#
# def crop_to_max_cube(mri_image_path, mask_image_path, output_image_path):
#     # 读取MRI影像和分割掩码
#     mri_image = sitk.ReadImage(mri_image_path)
#     mask_image = sitk.ReadImage(mask_image_path)
#
#     # 将分割掩码转换为NumPy数组
#     mask_array = sitk.GetArrayFromImage(mask_image)
#
#     # 找到分割掩码中非零值的最小和最大坐标
#     non_zero_indices = np.transpose(np.nonzero(mask_array))
#     min_coords = np.min(non_zero_indices, axis=0)
#     max_coords = np.max(non_zero_indices, axis=0)
#
#     # 计算包含分割区域的立方体的位置和大小
#     cube_start = list(map(int, min_coords))
#     cube_size = list(map(int, max_coords - min_coords + 1))
#
#     # 确保立方体区域在图像内
#     cube_start = np.maximum(cube_start, 0)
#     cube_size = np.minimum(cube_size, np.array(mri_image.GetSize()) - cube_start)
#
#     # 使用np.int64
#     cube_start = np.int64(cube_start)
#     cube_size = np.int64(cube_size)
#
#
#     # 使用RegionOfInterest方法裁剪MRI影像
#     cropped_image = sitk.RegionOfInterest(mri_image, list(cube_size), list(cube_start))
#
#     # 保存裁剪后的MRI影像
#     sitk.WriteImage(cropped_image, output_image_path)


def crop_image(image, pad_size, depth, bbox, save_path, save_name):
    roi_index = [x - y for x, y in zip(bbox[:3], [pad_size, pad_size, 0])]  # 角点坐标
    roi_size = [x + y for x, y in zip(bbox[3:], [pad_size * 2, pad_size * 2, depth])]  # ROI的宽度、高度和深度
    # print("roi_size: ", roi_size)
    image_crop = sitk.RegionOfInterest(image, roi_size, roi_index)
    cropped_image_save_path = save_path
    if not os.path.exists(cropped_image_save_path):
        os.mkdir(cropped_image_save_path)
    sitk.WriteImage(image_crop, os.path.join(cropped_image_save_path, save_name))
    return image_crop


def get_ROI_cube(image_file, mask_file):
    """
    读取图片
    Returns:

    """
    image = sitk.ReadImage(image_file)  # 读取图像文件
    mask_file = sitk.ReadImage(mask_file)  # 读取掩码文件
    seg_result_img = sitk.Cast(mask_file, sitk.sitkUInt8)
    seg_result_img.SetOrigin(image.GetOrigin())
    '''获取两个bounding box'''
    labelFilter = sitk.LabelShapeStatisticsImageFilter()
    labelFilter.Execute(seg_result_img)
    seg_result_bbox = labelFilter.GetBoundingBox(1)
    '''计算最终bounding box'''
    z_start = seg_result_bbox[2]
    z_end = seg_result_bbox[2] + seg_result_bbox[-1] - 1
    depth = z_end - z_start + 1 - seg_result_bbox[-1]
    bbox_list = list(seg_result_bbox)  # 元组不能改变内部元素，所以先将其转变为列表，改变元素值之后再转换为元组。
    bbox_list[2] = z_start
    bbox = tuple(bbox_list)

    pad_size = 20
    return pad_size, depth, bbox


def apply_roi(image, mask, save_path, save_name):
    """
    对于源目录下的所有3D图像，根据掩码文件ROI掩码获取最小的方框，并将其保存到目标目录下。
    """
    # 获取包围掩码ROI的最大3D方框
    pad_size, depth, bbox = get_ROI_cube(image, mask)

    # 裁剪3D图像
    cropped_image = crop_image(image, pad_size, depth, bbox, save_path, save_name)

    return cropped_image


def expand_image(image, pad_size, bbox, save_path, save_name):
    """

    Args:
        image:
        pad_size:
        bbox:
        save_path:
        save_name:

    Returns:

    """
    image = sitk.ReadImage(image)
    white_arr = np.zeros(image.GetSize()).T
    image_arr = sitk.GetArrayFromImage(image)
    image_shape = image_arr.shape
    origin = np.array([bbox[2], bbox[1] - pad_size, bbox[0] - pad_size])  # 将之前准备工作中获取的最终bbox的前三个元素作为参考原点。
    # 这里要注意，simpleITK和ndarray相互转换时其尺寸会相互颠倒，
    # simpleITK读取的图片尺寸格式为（w，h，d），转换成ndarray数组的尺寸会变为（d，h，w）
    print(origin)
    for z in range(image_shape[0]):
        for y in range(image_shape[1]):
            for x in range(image_shape[-1]):
                coordinate = np.array([z, y, x])
                mixed_coordinate = origin.__add__(coordinate)
                mixed_coordinate = tuple(mixed_coordinate)
                white_arr[mixed_coordinate] = image_arr[z, y, x]
    expanded_image = sitk.GetImageFromArray(white_arr)
    expanded_image.CopyInformation(image)
    expanded_image_save_path = save_path
    if not os.path.exists(expanded_image_save_path):
        os.mkdir(expanded_image_save_path)
    sitk.WriteImage(expanded_image, os.path.join(expanded_image_save_path, save_name))


def crop_by_intensity(img):
    xoy = np.sum(img, axis=2)
    yoz = np.sum(img, axis=0)
    xoz = np.sum(img, axis=1)

    ox = np.sum(xoy, axis=1)
    oy = np.sum(xoy, axis=0)
    oz = np.sum(xoz, axis=0)

    x1 = (ox != 0).argmax()
    x2 = (np.flipud(ox) != 0).argmax()
    x = ox.shape[0] - x1 - x2
    y1 = (oy != 0).argmax()
    y2 = (np.flipud(oy) != 0).argmax()
    y = oy.shape[0] - y1 - y2
    z1 = (oz != 0).argmax()
    z2 = (np.flipud(oz) != 0).argmax()
    z = oz.shape[0] - z1 - z2

    new_img = img[x1:x1 + x, y1:y1 + y, z1:z1 + z]

    return new_img


def crop_a(image, z):
    ls = []
    start = 0
    end = z
    for i in range(z):
        exist = (image[i, :, :] > 0) * 1
        # factor = np.ones(x, y)
        # res = np.dot(exist, factor)
        a = np.sum(exist)
        if a < 0:
            ls.append(0)
        else:
            ls.append(a)
    for i in range(len(ls)):
        if ls[i] != 0:
            start = i
            break
    for j in range(len(ls) - 1, 0, -1):
        if ls[j] != 0:
            end = j
            break
    return start, end


def crop_b(image, z):
    ls = []
    start = 0
    end = z
    for i in range(z):
        exist = (image[:, i, :] > 0) * 1
        # factor = np.ones(x, y)
        # res = np.dot(exist, factor)
        a = np.sum(exist)
        if a < 0:
            ls.append(0)
        else:
            ls.append(a)
    for i in range(len(ls)):
        if ls[i] != 0:
            start = i
            break
    for j in range(len(ls) - 1, 0, -1):
        if ls[j] != 0:
            end = j
            break
    return start, end


def crop_c(image, z):
    ls = []
    start = 0
    end = z
    for i in range(z):
        exist = (image[:, :, i] > 0) * 1
        # factor = np.ones(x, y)
        # res = np.dot(exist, factor)
        a = np.sum(exist)
        if a < 0:
            ls.append(0)
        else:
            ls.append(a)
    for i in range(len(ls)):
        if ls[i] != 0:
            start = i
            break
    for j in range(len(ls) - 1, 0, -1):
        if ls[j] != 0:
            end = j
            break
    return start, end


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    # spacing肯定不能是整数
    newSize = newSize.astype(np.int32)
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # ?
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def normalize_data(data):
    # b = np.percentile(data, 98)
    # t = np.percentile(data, 1)
    # data = np.clip(data,t,b)
    data = np.array(data, dtype=np.float32)
    means = data.mean()
    stds = data.std()
    # print(type(data),type(means),type(stds))
    data -= means
    data /= stds
    return data


def crop_roi(img_file, mask_file, save_file):
    img = sitk.ReadImage(img_file)
    mask = sitk.ReadImage(mask_file)

    spacing = img.GetSpacing()
    new_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # 根据预测的标签剪裁
    pre_imgarray = sitk.GetArrayFromImage(img)
    pre_maskarray = sitk.GetArrayFromImage(mask)
    (z, x, y) = pre_maskarray.shape
    start, end = crop_a(pre_maskarray, z)
    saveimg = pre_imgarray[start:end + 1, :, :]
    start, end = crop_b(pre_maskarray, x)
    saveimg = saveimg[:, start:end + 1, :]
    start, end = crop_c(pre_maskarray, y)
    saveimg = saveimg[:, :, start:end + 1]

    # resize
    saveimg = sitk.GetImageFromArray(saveimg)
    print(saveimg.GetSize())
    saveimg.SetDirection(new_direction)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetSpacing(spacing)
    resize_img = resize_image_itk(saveimg, (128, 128, 128), resamplemethod=sitk.sitkLinear)

    # 标准化
    resize_imgarr = sitk.GetArrayFromImage(resize_img)
    nor_resize_imgarr = normalize_data(resize_imgarr)
    nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)
    nor_resize_img.SetSpacing(resize_img.GetSpacing())
    nor_resize_img.SetOrigin(resize_img.GetOrigin())
    nor_resize_img.SetDirection(resize_img.GetDirection())

    sitk.WriteImage(nor_resize_img, save_file)




def crop_roi_expand(img_file, mask_file, save_file, expansion=5):
    img = sitk.ReadImage(img_file)
    mask = sitk.ReadImage(mask_file)

    spacing = img.GetSpacing()
    new_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # 根据预测的标签剪裁
    pre_imgarray = sitk.GetArrayFromImage(img)
    pre_maskarray = sitk.GetArrayFromImage(mask)
    (z, x, y) = pre_maskarray.shape
    start, end = crop_a(pre_maskarray, z)

    # Expand the crop region by 'expansion' pixels
    start = max(0, start - expansion)
    end = min(z - 1, end + expansion)

    saveimg = pre_imgarray[start:end + 1, :, :]
    start, end = crop_b(pre_maskarray, x)

    # Expand the crop region by 'expansion' pixels
    start = max(0, start - expansion)
    end = min(x - 1, end + expansion)

    saveimg = saveimg[:, start:end + 1, :]
    start, end = crop_c(pre_maskarray, y)

    # Expand the crop region by 'expansion' pixels
    start = max(0, start - expansion)
    end = min(y - 1, end + expansion)

    saveimg = saveimg[:, :, start:end + 1]

    # resize
    saveimg = sitk.GetImageFromArray(saveimg)
    # print(saveimg.GetSize())
    saveimg.SetDirection(new_direction)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetSpacing(spacing)
    resize_img = resize_image_itk(saveimg, (128, 128, 128), resamplemethod=sitk.sitkLinear)

    # 标准化
    resize_imgarr = sitk.GetArrayFromImage(resize_img)
    nor_resize_imgarr = normalize_data(resize_imgarr)
    nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)
    nor_resize_img.SetSpacing(resize_img.GetSpacing())
    nor_resize_img.SetOrigin(resize_img.GetOrigin())
    nor_resize_img.SetDirection(resize_img.GetDirection())

    sitk.WriteImage(nor_resize_img, save_file)


# 请确保在调用crop_roi函数时指定了expansion参数来控制向外扩展的距离


def batch_get_roi(img_dir, mask_dir, save_path):
    """
    把原来的图像和掩码图像分别裁剪ROI区域，然后再把裁剪后的图像和掩码图像分别扩展到原来的尺寸。
    Args:
        img_dir:
        mask_dir:
        save_path:

    Returns:

    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    patient_dirs = os.listdir(img_dir)
    for patient_dir in tqdm(patient_dirs):
        patient = os.path.join(img_dir, patient_dir)
        patient_mask = os.path.join(mask_dir, patient_dir + ".nii.gz")
        patient_save = os.path.join(save_path, patient_dir)
        if not os.path.exists(patient_save):
            os.mkdir(patient_save)
        for file in os.listdir(patient):
            if file.endswith(".nii.gz"):
                img_file = os.path.join(patient, file)
                save_file = os.path.join(patient_save, file)
                crop_roi_expand(img_file, patient_mask, save_file)


if __name__ == '__main__':
    # img_nii_gz_path = "./test_T1.nii.gz"
    # arr_img = load_nii_gz_as_array(img_nii_gz_path)
    # arr_img = random_augmentation(arr_img, (16, 16, 16), (0.75, 1.25), (0.7, 1.5), 1)
    # write_array_as_nii_gz(arr_img, "test.nii.gz")

    # get_max_3d_ROI("../test_data/Gliomas_00005_20181117", "../test_data_seg/Gliomas_00005_20181117.nii.gz",
    #                  "../test_data_out")
    # crop_to_max_cube("../test_data/Gliomas_00005_20181117/Gliomas_00005_20181117_T1.nii.gz",
    #                     "../test_data_seg/Gliomas_00005_20181117.nii.gz",
    #                     "../test_data_out/Gliomas_00005_20181117_T1.nii.gz")
    # roi = get_ROI_cube(sitk.ReadImage("../test_data/Gliomas_00005_20181117/Gliomas_00005_20181117_T1.nii.gz"),
    #                    "../test_data_seg/Gliomas_00005_20181117.nii.gz")
    # print(roi)
    # expand_image("../test_data/Gliomas_00005_20181117/Gliomas_00005_20181117_T1.nii.gz", roi[0], roi[2],
    #                 "../test_data_out", "test.nii.gz")
    # apply_roi("F:/Code/Medical/Glioma_easy/test_data/Gliomas_00005_20181117/Gliomas_00005_20181117_T1.nii.gz",
    #           "F:/Code/Medical/Glioma_easy/test_data_seg/Gliomas_00005_20181117.nii.gz",
    #           "F:/Code/Medical/Glioma_easy/test_data_out", "test.nii.gz")
    # img = crop_by_intensity(sitk.ReadImage("F:/Code/Medical/Glioma_easy/test_data_seg/Gliomas_00005_20181117.nii.gz")
    #                         )
    # sitk.WriteImage(img, "F:/Code/Medical/Glioma_easy/test_data_out/test2.nii.gz")
    # crop_roi("F:/Code/Medical/Glioma_easy/test_data/Gliomas_00005_20181117/Gliomas_00005_20181117_T1.nii.gz",
    #          "F:/Code/Medical/Glioma_easy/test_data_seg/Gliomas_00005_20181117.nii.gz",
    #          "F:/Code/Medical/Glioma_easy/test_data_out/test1.nii.gz")
    # crop_roi_expand("F:/Code/Medical/Glioma_easy/test_data/Gliomas_00005_20181117/Gliomas_00005_20181117_T1.nii.gz",
    #          "F:/Code/Medical/Glioma_easy/test_data_seg/Gliomas_00005_20181117.nii.gz",
    #          "F:/Code/Medical/Glioma_easy/test_data_out/test2.nii.gz")
    batch_get_roi("/media/spgou/DATA/ZYJ/Dataset/zscore_normalizedImages", "/media/spgou/DATA/ZYJ/Dataset/captk_before_data_net_seg",
                  "/media/spgou/DATA/ZYJ/Dataset/zscore_normalizedImages_ROI_images_expand")
