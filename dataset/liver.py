import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import SimpleITK as sitk
import numpy as np
from Get_Abdomen import config
from Get_Abdomen.models import pine_unet


# 不用生成标签 只需要把名字改为liver_0001.nii.gz
# 下采样固定spacing
def resize_image_by_spacing(itkimage, newSpacing, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSpacing = np.array(newSpacing, float)
    factor = originSpacing / newSpacing
    newSize = originSize * factor
    newSize = newSize.astype(np.int32)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def caijianx(image, x):
    startlist = []
    endlist = []
    for i in range(len(image)):
        for j in range(x):
            exist = (image[i, j, :] > 0) * 1
            a = np.sum(exist)
            if a > 0:
                startlist.append(j)
                break
        for j in range(x - 1, 0, -1):
            exist = (image[i, j, :] > 0) * 1
            a = np.sum(exist)
            if a > 0:
                endlist.append(j)
                break
    return min(startlist), max(endlist)


def caijiany(image, y):
    startlist = []
    endlist = []
    for i in range(len(image)):
        for j in range(y):
            exist = (image[i, :, j] > 0) * 1
            a = np.sum(exist)
            if a > 0:
                startlist.append(j)
                break
        for j in range(y - 1, 0, -1):
            exist = (image[i, :, j] > 0) * 1
            a = np.sum(exist)
            if a > 0:
                endlist.append(j)
                break
    return min(startlist), max(endlist)


def test(ct_array, model, mode=True):
    model.eval()
    data = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(1)
    # print(data.shape)
    output = model(data)
    if mode == True:
        output = torch.sigmoid(output)
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
    else:
        output = output.softmax(dim=1)
    output = output.squeeze(0)
    pred = torch.argmax(output, dim=0)
    # (96, 128, 128)
    a = torch.sum(pred.reshape(128 * 128 * 128), dim=0)
    pred = np.asarray(pred.cpu().numpy(), dtype='uint8')

    return pred


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
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


def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: truncated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


def get_label(out_img, pine_model):
    resize_img = resize_image_itk(out_img, (128, 128, 128), resamplemethod=sitk.sitkLinear)
    predLabel_array = test(sitk.GetArrayFromImage(resize_img), pine_model, False)
    predLabel = sitk.GetImageFromArray(predLabel_array)
    predLabel.SetOrigin(resize_img.GetOrigin())
    predLabel.SetDirection(resize_img.GetDirection())
    predLabel.SetSpacing(resize_img.GetSpacing())
    pre_mask = resize_image_by_spacing(predLabel, out_img.GetSpacing(), resamplemethod=sitk.sitkNearestNeighbor)
    return pre_mask


def jiancai_a(image, z):
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


def jiancai_b(image, z):
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


def jiancai_c(image, z):
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


def print_nii(saveimg, img):
    saveimg = sitk.GetImageFromArray(saveimg)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetDirection(img.GetDirection())
    saveimg.SetSpacing(img.GetSpacing())
    sitk.WriteImage(saveimg, "./img.nii.gz")


organ = "liver"
args = config.args
pine_model_path = '../Get_Abdomen/results/Inception'
pine_model_path_1 = "./"
# device = torch.device('cpu' if args.cpu else 'cuda')
pine_model = pine_unet.UNet(1, args.n_labels_seg, training=False)
ckpt = torch.load('{}/best_model.pth'.format(pine_model_path), map_location=torch.device('cpu'))
pine_model.load_state_dict(ckpt['net'])
save_path = r"../data/TrainingImg_1"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
path = r"../origin_data/liver/img"
file_list = os.listdir(path)
file_list.sort()
savename = []
for i in range(0, 10):
    savename.append("000" + str(i))
for i in range(10, 100):
    savename.append("00" + str(i))
for i in range(100, 1000):
    savename.append("0" + str(i))
num = 0
new_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
for file in file_list:
    print(file)
    img_path = os.path.join(path, file)
    img = sitk.ReadImage(img_path)
    spacing = img.GetSpacing()
    print(img.GetDirection())
    new_spacing = spacing
    img_fdata = sitk.GetArrayFromImage(img)
    img_fdata = np.flip(img_fdata, axis=1)
    name = file
    # 调窗
    ct_adjust = window_transform(img_fdata, 400, 40, normal=False)
    print_nii(ct_adjust, img)
    saveimg = ct_adjust

    # 获得腹部标签
    img_adjust = sitk.GetImageFromArray(saveimg)
    img_adjust.SetOrigin(img.GetOrigin())
    spacing = img.GetSpacing()
    img_adjust.SetSpacing(new_spacing)
    img_adjust.SetDirection(new_direction)

    pre_mask = get_label(img_adjust, pine_model)

    pre_maskarray = sitk.GetArrayFromImage(pre_mask)
    (z, x, y) = pre_maskarray.shape
    start, end = jiancai_b(pre_maskarray, x)
    saveimg = saveimg[:, start:end + 1, :]
    start, end = jiancai_c(pre_maskarray, y)
    saveimg_fdata = saveimg[:, :, start:end + 1]

    # resize
    saveimg = sitk.GetImageFromArray(saveimg_fdata)
    saveimg.SetSpacing(new_spacing)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetDirection(new_direction)

    pine_model_1 = pine_unet.UNet(1, 4, training=False)
    ckpt_1 = torch.load('{}/best_model.pth'.format(pine_model_path_1), map_location=torch.device('cpu'))
    pine_model_1.load_state_dict(ckpt_1['net'])
    pre_3organ = get_label(saveimg, pine_model_1)
    pre_maskarray = sitk.GetArrayFromImage(pre_3organ)
    # 根据预测的标签剪裁
    (z, x, y) = pre_maskarray.shape
    start, end = jiancai_a(pre_maskarray, z)
    saveimg = saveimg_fdata[start:end + 1, :, :]
    start, end = jiancai_b(pre_maskarray, x)
    saveimg = saveimg[:, start:end + 1, :]
    start, end = jiancai_c(pre_maskarray, y)
    saveimg = saveimg[:, :, start:end + 1]

    # resize
    saveimg = sitk.GetImageFromArray(saveimg)
    saveimg.SetSpacing(new_spacing)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetDirection(new_direction)
    resize_img = resize_image_itk(saveimg, (128, 128, 128), resamplemethod=sitk.sitkLinear)

    # 标准化
    resize_imgarr = sitk.GetArrayFromImage(resize_img)
    nor_resize_imgarr = normalize_data(resize_imgarr)
    nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)
    nor_resize_img.SetSpacing(resize_img.GetSpacing())
    nor_resize_img.SetOrigin(resize_img.GetOrigin())
    nor_resize_img.SetDirection(resize_img.GetDirection())

    # 保存
    sitk.WriteImage(nor_resize_img, os.path.join(save_path, organ + "_" + savename[num] + ".nii.gz"))
    num = num + 1
