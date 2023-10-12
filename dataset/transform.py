from PIL import Image, ImageFilter
from torchvision import transforms
from scipy.ndimage import zoom
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import random
import tqdm
from pathlib import Path


class MedicalImageScaler:
    def __init__(self, scale_factors):
        """
        参数：
            scale_factors：一个长度为3的元组或列表，表示在每个维度上的缩放因子。
        """
        self.scale_factors = scale_factors

    def ensure_valid_shape(self, image):
        """
        确保影像满足缩放的要求，若不满足，则进行裁剪或填充。

        参数：
            image：numpy数组，形状为(D, H, W)，表示3D医学影像数据。

        返回：
            new_image：numpy数组，形状为(D', H', W')，表示处理后的3D医学影像数据。
        """
        original_shape = image.shape
        target_shape = [int(dim * factor) for dim, factor in zip(image.shape, self.scale_factors)]

        # 如果原始影像尺寸等于目标尺寸，则无需处理
        if original_shape == target_shape:
            return image

        # 将目标尺寸限制在原始影像尺寸范围内，避免缩放后尺寸过小或过大
        target_shape = [min(dim, target_dim) for dim, target_dim in zip(original_shape, target_shape)]

        # 使用裁剪或填充来调整影像尺寸
        new_image = np.zeros(target_shape, dtype=image.dtype)
        slices = tuple(slice(0, min(dim, target_dim)) for dim, target_dim in zip(original_shape, target_shape))
        new_image[slices] = image[slices]

        return new_image

    def __call__(self, image):
        """
        参数：
            image：numpy数组，形状为(D, H, W)，表示3D医学影像数据。

        返回：
            scaled_image：numpy数组，形状为(D', H', W')，表示缩放后的3D医学影像数据。
        """
        # 确保影像满足缩放的要求
        image = self.ensure_valid_shape(image)

        # 使用SciPy的zoom函数进行缩放
        scaled_image = zoom(image, self.scale_factors, order=3)  # 使用三次样条插值进行缩放

        return scaled_image


#
#
# # 随机旋转
# class RandomRotate(object):
#     def __init__(self, degree, p=0.5):
#         self.degree = degree
#         self.p = p
#
#     def __call__(self, img):
#         if random.random() < self.p:
#             rotate_degree = random.uniform(-1 * self.degree, self.degree)
#             img = img.rotate(rotate_degree, Image.BILINEAR)
#         return img
#
#
# # 随机高斯模糊
class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img


# 随机垂直翻转
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


# 随机水平翻转
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


# 弹性变换
class ElasticTransform(object):
    def __init__(self, alpha=1, sigma=50, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            shape = img.shape
            dx = np.random.uniform(-1, 1, shape) * self.alpha
            dy = np.random.uniform(-1, 1, shape) * self.alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            img = np.stack(
                [scipy.ndimage.interpolation.map_coordinates(channel, indices, order=1).reshape(shape) for channel in
                 img], axis=2)
            img = Image.fromarray(img.astype(np.uint8))
        return img


def resize(img, shape, mode='constant', orig_shape=(240, 240, 155)):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[-3] / orig_shape[0],
        shape[-2] / orig_shape[1],
        shape[-1] / orig_shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode)


class Resize(object):
    def __init__(self, shape, mode='constant', orig_shape=(128, 128, 128)):
        self.shape = shape
        self.mode = mode
        self.orig_shape = orig_shape

    def __call__(self, img):
        if len(img.shape) == 3:
            img = resize(img, self.shape, self.mode, self.orig_shape)
        elif len(img.shape) == 4:
            # 按照第0个维度分别对每个通道进行resize
            img = np.stack(
                [resize(img[i, :, :, :], self.shape, self.mode, self.orig_shape) for i in range(img.shape[0])], axis=0)
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
        return img


def random_augmentation(arr_img, shift_range, scale_range, gamma_range, p):
    arr_img = random_gamma_transformation(arr_img, gamma_range, 0.15)
    arr_img = random_flip(arr_img, p)
    arr_img = random_permute(arr_img, p)
    arr_img = random_rotate(arr_img, angle_range=30, p=0.2)
    arr_img = random_shift(arr_img, shift_range, p)
    # arr_img = random_scale(arr_img, scale_range, p)
    arr_img = random_noise(arr_img, 0.15)
    return arr_img


def patch_random_augmentation(arr_img, shift_range, scale_range, gamma_range, p):
    if random.random() < p:
        arr_img = random_gamma_transformation(arr_img, gamma_range)
        arr_img = random_flip(arr_img, p)
        arr_img = random_rotate(arr_img, 3, p)
        arr_img = random_shift(arr_img, shift_range, p)
        # arr_img = random_scale(arr_img, scale_range, p)
        arr_img = random_noise(arr_img, p)
    return arr_img


def random_rotate(arr_img, angle_range, p):
    if random.random() < p:
        angle_x, angle_y, angle_z = random.uniform(-angle_range, angle_range), random.uniform(-angle_range,
                                                                                              angle_range), \
            random.uniform(-angle_range, angle_range)
        arr_img = rotate(arr_img, angle_x, "x", order=3)
        arr_img = rotate(arr_img, angle_y, "y", order=3)
        arr_img = rotate(arr_img, angle_z, "z", order=3)
    return arr_img


def rotate(arr_img, angle, axial, order=3):
    if order == 0:
        cval = 0
    else:
        cval = np.percentile(arr_img, 1)
    if axial == "z":
        arr_img = scipy.ndimage.rotate(arr_img, angle, (1, 2), reshape=False, cval=cval, order=order)
    elif axial == "y":
        arr_img = scipy.ndimage.rotate(arr_img, angle, (0, 2), reshape=False, cval=cval, order=order)
    elif axial == "x":
        arr_img = scipy.ndimage.rotate(arr_img, angle, (0, 1), reshape=False, cval=cval, order=order)
    else:
        raise ValueError("axial must be one of x, y or z")
    return arr_img


def random_shift(arr_img, shift_range, p):
    if random.random() < p:
        shift_z_range, shift_y_range, shift_x_range = shift_range
        shift_x, shift_y, shift_z = random.uniform(-shift_x_range, shift_x_range), \
            random.uniform(-shift_y_range, shift_y_range), \
            random.uniform(-shift_z_range, shift_z_range)
        shift_zyx = (shift_z, shift_y, shift_x)
        arr_img = shift_img(arr_img, shift_zyx, order=3)
    return arr_img


def shift_img(arr_img, shift_zyx, order):
    if len(arr_img.shape) == 3:
        if order == 0:
            cval = 0
        else:
            cval = np.percentile(arr_img, 1)
        arr_img = scipy.ndimage.shift(arr_img, shift=shift_zyx, cval=cval, order=order)
        return arr_img
    elif len(arr_img.shape) == 4:
        arr_img = np.stack([shift_img(arr_img[i, :, :, :], shift_zyx, order) for i in range(arr_img.shape[0])], axis=0)
        return arr_img


def random_flip(arr_img, p):
    if random.random() < p:
        axial_list = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        axial = random.choice(axial_list)
        for sub_axial in axial:
            arr_img = flip(arr_img, sub_axial)
    return arr_img


def flip(arr_img, axis):
    if len(arr_img.shape) == 3:
        arr_img = np.flip(arr_img, axis=axis)
    elif len(arr_img.shape) == 4:
        for i in range(arr_img.shape[0]):
            arr_img[i, :, :, :] = flip(arr_img[i, :, :, :], axis)
    return arr_img


def random_permute(arr_img, p):
    if random.random() < p:
        axial_list = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        permution = random.choice(axial_list)
        arr_img = permute(arr_img, permution)
    return arr_img


def permute(arr_img, permution):
    if len(arr_img.shape) == 3:
        arr_img = np.transpose(arr_img, axes=permution)
    elif len(arr_img.shape) == 4:
        for i in range(arr_img.shape[0]):
            arr_img[i, :, :, :] = permute(arr_img[i, :, :, :], permution)
    # arr_img = np.transpose(arr_img, axes=permution)
    return arr_img


def random_noise(arr_image, p):
    if random.random() < p:
        # arr_image = noise(arr_image)
        arr_image = gaussian_noise(arr_image)
    return arr_image


def std_noise(arr_image):
    if len(arr_image == 3):
        std = np.std(arr_image)
        noise = np.random.random(arr_image.shape)
        noise = 0.1 * std * 2 * (noise - 0.5)
        arr_image = arr_image + noise
        return arr_image
    elif len(arr_image == 4):
        for i in range(arr_image.shape[0]):
            arr_image[i, :, :, :] = std_noise(arr_image[i, :, :, :])
        return arr_image
    else:
        raise ValueError("arr_image must be 3D or 4D")


def gaussian_noise(arr_image):
    # 使用高斯噪声，设置均值为0, 方差为0到0.1的均匀分布
    if len(arr_image.shape) == 3:
        return arr_image + np.random.normal(0, np.random.uniform(0, 0.1), arr_image.shape)
    elif len(arr_image.shape) == 4:
        for i in range(arr_image.shape[0]):
            arr_image[i, :, :, :] = gaussian_noise(arr_image[i, :, :, :])
        return arr_image
    else:
        raise ValueError("arr_image must be 3D or 4D")
    # return arr_image + np.random.normal(0, np.random.uniform(0, 0.1), arr_image.shape)



def random_scale(arr_image, scale_factor_range, p):
    if random.random() < p:
        low, hi = scale_factor_range
        scale_factor = random.uniform(low, hi)
        arr_image = scale(arr_image, scale_factor, order=3)
    return arr_image


def scale(arr_image, scale_factor, order=3):
    shapes = arr_image.shape
    if order == 0:
        cval = 0
    else:
        cval = np.percentile(arr_image, 1)
    arr_image = scipy.ndimage.zoom(arr_image, zoom=scale_factor, order=order, mode="constant", cval=cval)
    scaled_shapes = arr_image.shape
    for i, zip_data in enumerate(zip(shapes, scaled_shapes)):
        shape, scaled_shape = zip_data
        if scaled_shape < shape:
            padding = shape - scaled_shape
            left_padding = padding // 2
            right_padding = padding - left_padding
            if i == 0:
                arr_image = np.pad(arr_image, ((left_padding, right_padding), (0, 0),
                                               (0, 0)), constant_values=cval)
            elif i == 1:
                arr_image = np.pad(arr_image, ((0, 0), (left_padding, right_padding),
                                               (0, 0)), constant_values=cval)
            elif i == 2:
                arr_image = np.pad(arr_image, ((0, 0), (0, 0),
                                               (left_padding, right_padding)), constant_values=cval)
        elif scaled_shape > shape:
            crop = scaled_shape - shape
            left_crop = crop // 2
            right_crop = crop - left_crop
            if i == 0:
                arr_image = arr_image[left_crop: scaled_shape - right_crop, :, :]
            elif i == 1:
                arr_image = arr_image[:, left_crop: scaled_shape - right_crop, :]
            elif i == 2:
                arr_image = arr_image[:, :, left_crop: scaled_shape - right_crop]
    return arr_image


def random_gamma_transformation(arr_image, gamma_range=(0.7, 1.5), p=0.15):
    if random.random() < p:
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        arr_image = gamma_transformation(arr_image, gamma)
    return arr_image


def gamma_transformation(arr_image, gamma):
    if len(arr_image.shape) == 3:
        low, hi = arr_image.min(), arr_image.max()
        arr_image = (arr_image - low) / (hi - low)
        arr_image = np.power(arr_image, gamma)
        arr_image = (hi - low) * arr_image + low
        return arr_image
    elif len(arr_image.shape) == 4:
        for i in range(arr_image.shape[0]):
            arr_image[i, :, :, :] = gamma_transformation(arr_image[i, :, :, :], gamma)
        return arr_image


def load_nii_gz_as_array(nii_gz_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(nii_gz_path))


def write_array_as_nii_gz(arr_img, out_path):
    sitk.WriteImage(sitk.GetImageFromArray(arr_img), out_path)


class Scale(object):
    """
    把scale函数封装成类，方便transform.Compose使用
    """

    def __init__(self, scale_factor, order=3):
        self.scale_factor = scale_factor
        self.order = order

    def __call__(self, arr_img):
        return scale(arr_img, self.scale_factor, order=self.order)


class RandomScale(object):
    """
    把random_scale函数封装成类，方便transform.Compose使用
    """

    def __init__(self, scale_range, order):
        self.scale_range = scale_range
        self.order = order

    def __call__(self, arr_img):
        return random_scale(arr_img, self.scale_range, self.order)


# class RandomGammaTransformation(object):
#     """
#     把random_gamma_transformation函数封装成类，方便transform.Compose使用
#     """
#
#     def __init__(self, gamma_range):
#         self.gamma_range = gamma_range
#
#     def __call__(self, arr_img):
#         return random_gamma_transformation(arr_img, self.gamma_range)


# class RandomFlip(object):
#     """
#     把random_flip函数封装成类，方便transform.Compose使用
#     """
#
#     def __init__(self):
#         pass
#
#     def __call__(self, arr_img):
#         return random_flip(arr_img)


# class RandomPermute(object):
#     """
#     把random_permute函数封装成类，方便transform.Compose使用
#     """
#
#     def __init__(self):
#         pass
#
#     def __call__(self, arr_img):
#         return random_permute(arr_img)


class RandomRotate(object):
    """
    把random_rotate函数封装成类，方便transform.Compose使用
    """

    def __init__(self, angle_range):
        self.angle_range = angle_range

    def __call__(self, arr_img):
        return random_rotate(arr_img, self.angle_range)


class RandomShift(object):
    """
    把random_shift函数封装成类，方便transform.Compose使用
    """

    def __init__(self, shift_range):
        self.shift_range = shift_range

    def __call__(self, arr_img):
        return random_shift(arr_img, self.shift_range)


class RandomNoise(object):
    """
    把random_noise函数封装成类，方便transform.Compose使用
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, arr_img):
        return random_noise(arr_img, p=self.p)


class RandomAugmentation(object):
    """
    把random_augmentation函数封装成类，方便transform.Compose使用
    """

    def __init__(self, shift_range, scale_range, gamma_range):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.gamma_range = gamma_range

    def __call__(self, arr_img):
        return random_augmentation(arr_img, self.shift_range, self.scale_range, self.gamma_range, p=0.15)


class PatchRandomAugmentation(object):
    """
    把patch_random_augmentation函数封装成类，方便transform.Compose使用
    """

    def __init__(self, shift_range, scale_range, gamma_range):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.gamma_range = gamma_range

    def __call__(self, arr_img):
        return patch_random_augmentation(arr_img, self.shift_range, self.scale_range, self.gamma_range)


# if __name__ == '__main__':
#     img_nii_gz_path = "./test_T1.nii.gz"
#     arr_img = load_nii_gz_as_array(img_nii_gz_path)
#     arr_img = scale(arr_img, 1.25, 3)
#     arr_img = random_permute(arr_img, 1)
#     arr_img = random_flip(arr_img, 1)
#     arr_img = random_shift(arr_img, (16, 16, 16), 1)
#     arr_img = random_augmentation(arr_img,  (16, 16, 16), (0.75, 1.25), (0.7, 1.5), 1)
#     write_array_as_nii_gz(arr_img, "test.nii.gz")


## nnU-Net的图像增强方法
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor


# GaussianNoiseTransform(p_per_sample=0.1),
# GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
#                      p_per_channel=0.5),
# BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
# ContrastAugmentationTransform(p_per_sample=0.15),
# SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
#                               p_per_channel=0.5,
#                               order_downsample=0, order_upsample=3, p_per_sample=0.25,
#                               ignore_axes=ignore_axes),
# GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),
# GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3),
# MirrorTransform(axes),

class GaussianNoise(AbstractTransform):
    def __init__(self, p_per_sample=0.1):
        self.p_per_sample = p_per_sample

    def __call__(self, img):
        if len(img.shape) == 3:
            img = GaussianNoiseTransform(p_per_sample=1)()(img)
        elif len(img.shape) == 4:
            img = np.stack([GaussianNoiseTransform(p_per_sample=1)()(img[i, :, :, :]) for i in range(img.shape[0])],
                           axis=0)
        else:
            raise ValueError("Unsupported shape")
        return img


class GaussianBlur(AbstractTransform):
    def __init__(self, p_per_sample=0.1):
        self.p_per_sample = p_per_sample

    def __call__(self, img):
        if len(img.shape) == 3:
            img = GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=1,
                                        p_per_channel=0.5)(img)
        elif len(img.shape) == 4:
            img = np.stack([GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=1,
                                                  p_per_channel=0.5)(img[i, :, :, :]) for i in range(img.shape[0])],
                           axis=0)
        else:
            raise ValueError("Unsupported shape")
        return img


class BrightnessMultiplicative(AbstractTransform):
    def __init__(self, p_per_sample=0.1):
        self.p_per_sample = p_per_sample

    def __call__(self, img):
        if len(img.shape) == 3:
            img = BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=1)(img)
        elif len(img.shape) == 4:
            img = np.stack(
                [BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=1)(img[i, :, :, :]) for i
                 in range(img.shape[0])], axis=0)
        else:
            raise ValueError("Unsupported shape")
        return img


class ContrastAugmentation(AbstractTransform):
    def __init__(self, p_per_sample=0.1):
        self.p_per_sample = p_per_sample

    def __call__(self, img):
        if len(img.shape) == 3:
            img = ContrastAugmentationTransform(p_per_sample=1)(img)
        elif len(img.shape) == 4:
            img = np.stack(
                [ContrastAugmentationTransform(p_per_sample=1)(img[i, :, :, :]) for i in range(img.shape[0])], axis=0)
        else:
            raise ValueError("Unsupported shape")
        return img


class SimulateLowResolution(AbstractTransform):
    def __init__(self, p_per_sample=0.1):
        self.p_per_sample = p_per_sample

    def __call__(self, img):
        if len(img.shape) == 3:
            img = SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                 p_per_channel=0.5,
                                                 order_downsample=0, order_upsample=3, p_per_sample=1,
                                                 ignore_axes=None)(img)
        elif len(img.shape) == 4:
            img = np.stack([SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                           p_per_channel=0.5,
                                                           order_downsample=0, order_upsample=3, p_per_sample=1,
                                                           ignore_axes=None)(img[i, :, :, :]) for i in
                            range(img.shape[0])], axis=0)
        else:
            raise ValueError("Unsupported shape")
        return img


class Gamma(AbstractTransform):
    def __init__(self, p_per_sample=0.1, gamma_range=(0.7, 1.5), invert_image=True, per_channel=True,
                 retain_stats=True):
        self.p_per_sample = p_per_sample
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    def __call__(self, img):
        if len(img.shape) == 3:
            img = GammaTransform(self.gamma_range, self.invert_image, self.per_channel, self.retain_stats,
                                 p_per_sample=self.p_per_sample)(img)
        elif len(img.shape) == 4:
            img = np.stack([GammaTransform(self.gamma_range, self.invert_image, self.per_channel, self.retain_stats,
                                           p_per_sample=self.p_per_sample)(img[i, :, :, :]) for i in
                            range(img.shape[0])], axis=0)
        else:
            raise ValueError("Unsupported shape")
        return img


class Mirror(AbstractTransform):
    def __init__(self, axes):
        self.axes = axes

    def __call__(self, img):
        if len(img.shape) == 3:
            img = MirrorTransform(self.axes)(img)
        elif len(img.shape) == 4:
            img = np.stack([MirrorTransform(self.axes)(img[i, :, :, :]) for i in range(img.shape[0])], axis=0)
        else:
            raise ValueError("Unsupported shape")
        return img
