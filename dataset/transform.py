import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import Resize
from scipy.ndimage import zoom

# 缩放
# class Resize(object):
#     def __init__(self, size, interpolation=Image.BILINEAR):
#         self.size = size
#         self.interpolation = interpolation
#
#     def __call__(self, img):
#         # padding
#         ratio = self.size[0] / self.size[1]
#         w, h = img.size
#         if w / h < ratio:
#             t = int(h * ratio)
#             w_padding = (t - w) // 2
#             img = img.crop((-w_padding, 0, w+w_padding, h))
#         else:
#             t = int(w / ratio)
#             h_padding = (t - h) // 2
#             img = img.crop((0, -h_padding, w, h+h_padding))
#
#         img = img.resize(self.size, self.interpolation)
#
#         return img

# 3D医学影像的缩放
class Resize3D(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


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


# # 随机裁剪
# class RandomCrop(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, img):
#         w, h = img.size
#         th, tw = self.size
#
#         if w == tw and h == th:
#             return img
#
#         x1 = random.randint(0, w - tw)
#         y1 = random.randint(0, h - th)
#         img = img.crop((x1, y1, x1 + tw, y1 + th))
#
#         return img


# 随机旋转
class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img


# 随机高斯模糊
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


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def train_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        RandomRotate(15, 0.3),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def val_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
