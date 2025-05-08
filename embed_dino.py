from typing import Iterable
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter, Image
import einops
import os
import matplotlib.pyplot as plt
from preprocess import keep_only_breast
import pandas as pd
from tqdm import tqdm
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from dinov2.models.vision_transformer import vit_base

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
        
    def pad(self, image, ar=1):
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
        
class Transform(object):
    def __init__(
        self, is_train: bool = True, img_size: int = 518
    ):
        self.data_transforms = transforms.Compose(
            [
                KeepOnlyBreast(),
                OtsuCut(),
                Pad(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.118, 0.118, 0.118], [0.1775, 0.1775, 0.1775]),
            ]
        )

    def __call__(self, img):
        return self.data_transforms(img)


class EmbedDataset(Dataset):
    def __init__(self, split, transform):
        df = pd.read_csv(f'{split}.csv')
        df = df.dropna(subset=['asses', 'png_path'])

        df_5 = df[df['asses'] == 5.0]
        df_1 = df[df['asses'] == 1.0].sample(len(df_5), random_state=42)

        self.df = pd.concat([df_5, df_1]).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open('/data/' + row['png_path'])
        img_tensor = self.transform(img)
        label = row['asses']
        return img_tensor, label

def embed(split, ckpt_path, suffix):
    dataset = EmbedDataset(split, Transform())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=64)

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to('cuda')

    embeddings = []
    labels = []

    for batch_imgs, batch_labels in tqdm(dataloader):
        try:
            batch_imgs = batch_imgs.to('cuda')
            with torch.no_grad():
                batch_embeddings = model(batch_imgs)
            embeddings.extend(batch_embeddings.cpu().numpy())
            labels.extend(batch_labels.numpy())
        except Exception as e:
            print(f"Batch failed: {e}")

    np.savez(f'{split}_{suffix}.npz', embeddings=np.array(embeddings), labels=np.array(labels))



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python embed.py <ckpt_path> <suffix>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    suffix = sys.argv[2]

    embed('train', ckpt_path, suffix)
    embed('valid', ckpt_path, suffix)
    embed('test', ckpt_path, suffix)
