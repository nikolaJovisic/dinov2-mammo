import logging
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import random
from torchvision.transforms.functional import crop, resize

logger = logging.getLogger("dinov2")

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class GrayscaleToRGBTransform:
    def __call__(self, img):
        img_np = np.array(img)
        if img_np.ndim == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np
        return Image.fromarray(img_rgb)

class RandomRightAngleRotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return img.rotate(angle)

class CustomRandomCrop:
    def __init__(self, size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC, threshold=0.2, max_tries=50):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.interpolation = interpolation
        self.threshold = threshold
        self.max_tries = max_tries

    def __call__(self, img):
        img_w, img_h = img.size
        min_dim = min(img_w, img_h)
        base_area = min_dim ** 2

        for _ in range(self.max_tries):
            target_area = base_area * torch.empty(1).uniform_(*self.scale).item()
            crop_size = int(round(target_area ** 0.5))

            if crop_size <= min_dim:
                if img_w > img_h:
                    top = torch.randint(0, img_h - crop_size + 1, (1,)).item()
                    left = torch.randint(0, img_w - crop_size + 1, (1,)).item()
                else:
                    top = torch.randint(0, img_h - crop_size + 1, (1,)).item()
                    left = torch.randint(0, img_w - crop_size + 1, (1,)).item()

                cropped = crop(img, top, left, crop_size, crop_size)
                resized = resize(cropped, self.size, interpolation=self.interpolation)

                if np.mean(np.array(resized)) / 255 > self.threshold:
                    return resized

        return resize(img, self.size, interpolation=self.interpolation)



class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=518,
        local_crops_size=98,
        normalization=None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.geometric_augmentation_global = transforms.Compose(
            [
                CustomRandomCrop(global_crops_size, scale=global_crops_scale),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                CustomRandomCrop(local_crops_size, scale=local_crops_scale),
                RandomRightAngleRotation(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        global_transfo2_extra = transforms.Compose(
            [
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                AddGaussianNoise(std=0.01),
                transforms.Normalize(*normalization),
            ]
        )

        self.to_rgb = GrayscaleToRGBTransform()

        self.global_transfo1 = transforms.Compose([self.to_rgb, self.normalize])
        self.global_transfo2 = transforms.Compose([self.to_rgb, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([self.to_rgb, self.normalize])

    def __call__(self, image):
        image = image.convert("L")

        output = {}

        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output, None
