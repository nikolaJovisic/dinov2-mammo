from typing import Iterable
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter, Image
import einops
import os
import matplotlib.pyplot as plt
from .preprocess import keep_only_breast
import pandas as pd
from tqdm import tqdm
import torch
import random
from torch.utils.data import Dataset

def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


class OtsuCut(object):

    def __init__(self):
        super().__init__()

    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        mask = otsu_mask(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))
        # Convert to NumPy array if not already

        # Check if the matrix is empty or has no '1's
        if mask.size == 0 or not np.any(mask):
            return Image.fromarray(x)

        # Find the rows and columns where '1' appears
        rows = np.any(mask == 255, axis=1)
        cols = np.any(mask == 255, axis=0)

        # Find the indices of the rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        # Crop and return the submatrix
        x = x[min_row : max_row + 1, min_col : max_col + 1]
        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)
        
class KeepOnlyBreast(object):
    def __init__(self):
        super().__init__()


    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
            
        x, _ = keep_only_breast(x)
        x = einops.repeat(x, 'h w -> h w 3')
        
        x = (x // 255).astype(np.uint8)
        
        
        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)
        
class Pad(object):
    def __init__(self):
        super().__init__()
        
    def flip_if_should(self, image):
        x_center = image.shape[1] // 2
        col_sum = image.sum(axis=0)

        left_sum = np.sum(col_sum[0:x_center])
        right_sum = np.sum(col_sum[x_center:-1])
        

        if left_sum < right_sum:
            return np.fliplr(image)
        else:
            return image
        
        
    def pad(self, image, ar=1):
        image = self.flip_if_should(image)
        n_rows, n_cols = image.shape[:2]
        image_ratio = n_rows / n_cols
        if image_ratio == ar:
            return image
        if ar < image_ratio:
            new_n_cols = int(n_rows / ar)
            ret_val = np.zeros((n_rows, new_n_cols, image.shape[2]), dtype=image.dtype)
        else:
            new_n_rows = int(n_cols * ar)
            ret_val = np.zeros((new_n_rows, n_cols, image.shape[2]), dtype=image.dtype)
        ret_val[:n_rows, :n_cols] = image
        return ret_val


    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
            
        x = self.pad(x)
        
        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)
        
class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)
        
class CustomRandomCrop:
    def __init__(self, size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC, threshold=0.2, max_tries=10):
        self.size = size
        self.scale = scale
        self.interpolation = interpolation
        self.threshold = threshold
        self.max_tries = max_tries

    def __call__(self, img):
        for _ in range(self.max_tries):
            crop = transforms.RandomResizedCrop(self.size, scale=self.scale, interpolation=self.interpolation)(img)
            if np.average(np.array(crop)) / 255 > self.threshold:
                return crop
        return crop
        
        
class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=518,
        local_crops_size=98,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                CustomRandomCrop(
                    global_crops_size, scale=global_crops_scale
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                CustomRandomCrop(
                    local_crops_size, scale=local_crops_scale
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.0)],
                    p=0.8,
                ),
                #transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.118, 0.118, 0.118], [0.1775, 0.1775, 0.1775]),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
        
class Transform(object):
    def __init__(self):
        self.data_transforms = transforms.Compose(
            [
                KeepOnlyBreast(),
                OtsuCut(),
                Pad(),
                DataAugmentationDINO(global_crops_scale=(0.5, 1.0), local_crops_scale=(0.01, 0.1), local_crops_number=8)
            ]
        )

    def __call__(self, img):
        return self.data_transforms(img)
    
class EmbedDataset(Dataset):
    def __init__(self, csv_path: str, root: str, transform=Transform()):
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.iloc[idx]['png_path'])
        
        image = Image.open(img_path)
        image = self.transform(image)

        return image, None