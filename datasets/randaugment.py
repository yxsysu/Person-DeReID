import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torchvision import transforms
# from .vpd_augmentation import return_strong_aug
from timm.data.random_erasing import RandomErasing
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool(cfg):
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool(cfg):
    # FixMatch paper
    augs = [
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        # (Posterize, 4, 4),
        # (Rotate, 30, 0),
        # (Sharpness, 0.9, 0.05),
        # (ShearX, 0.3, 0),
        # (ShearY, 0.3, 0),
        # (Solarize, 256, 0),
        # (TranslateX, 0.3, 0),
        # (TranslateY, 0.3, 0)]
    ]
    return augs



# class RandAugmentPC(object):
#     def __init__(self, n, m):
#         assert n >= 1
#         assert 1 <= m <= 10
#         self.n = n
#         self.m = m
#         self.augment_pool = my_augment_pool()
#
#     def __call__(self, img):
#         ops = random.choices(self.augment_pool, k=self.n)
#         for op, max_v, bias in ops:
#             prob = np.random.uniform(0.2, 0.8)
#             if random.random() + prob >= 1:
#                 img = op(img, v=self.m, max_v=max_v, bias=bias)
#         img = CutoutAbs(img, int(32*0.5))
#         return img


class RandAugmentMC(object):
    def __init__(self, n, m, cfg):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        if cfg.TGT_DATA.MY_AUG_POOL:
            self.augment_pool = my_augment_pool(cfg)
        else:
            self.augment_pool = fixmatch_augment_pool(cfg)

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        # for op, _, _ in ops:
        #     if isinstance(op, RandomizedQuantizationAugModule):
        #         img = op(img)
        #         return img
        # candidate = np.array([0, 0])
        # while candidate.sum() == 0:
        #     candidate = np.random.choice(np.arange(2), 2, replace=True)
        # count = 0
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            # if candidate[count] > 0:
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
            # count = count + 1
        img = CutoutAbs(img, int(32*0.5))
        return img


class MutualTransform(object):
    def __init__(self, cfg):
        self.weak = transforms.Compose([
            transforms.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            transforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            transforms.Pad(cfg.INPUT.PADDING),
            transforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])

        self.strong = transforms.Compose([
            transforms.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            transforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            transforms.Pad(cfg.INPUT.PADDING),
            transforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandAugmentMC(n=2, m=10, cfg=cfg)]
        )
        self.vpd = cfg.TGT_DATA.VPD
        if self.vpd > 0.0:
            print('apply vpd aug with prob ', self.vpd)
        # region_num = 8
        # self.quant_layer = transforms.Compose([
        #     transforms.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        #     transforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        #     transforms.Pad(cfg.INPUT.PADDING),
        #     transforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        #     transforms.ToTensor(),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
        #     transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
        # self.vpd_aug = return_strong_aug(cfg)


    def __call__(self, x):
        weak = self.weak(x)

        # if random.random() < self.vpd:
        #     strong = self.vpd_aug(x)
        # else:
        #
        strong = self.strong(x)
        weak = self.normalize(weak)
        strong = self.normalize(strong)
        return weak, strong


class RandomizedQuantizationAugModule(nn.Module):
    def __init__(self, region_num, collapse_to_val='inside_random', spacing='random',
                 transforms_like=False, p_random_apply_rand_quant = 1):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant

    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0] # min, max over batch size, spatial dimension
        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c
            x = x.view(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) # -> (C), (C), (C)

        # region percentiles for each channel
        if self.spacing == "random":
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            region_percentiles = torch.tile(torch.arange(1/(total_region_percentile_number_per_channel[0] + 1), 1, step=1/(total_region_percentile_number_per_channel[0]+1), device=x.device), [C])
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])
        # ordered region ends
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        # ordered middle points
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2

        # associate region id
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)

        if self.collapse_to_val == 'middle':
            # middle points as the proxy for all values in corresponding regions
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            # random points inside each region as the proxy for all values in corresponding regions
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(), device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.region_num])
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (ordered_region_right_ends - ordered_region_left_ends)
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = torch.where(torch.rand([B,1,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = torch.where(torch.rand([C,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)

        return x


