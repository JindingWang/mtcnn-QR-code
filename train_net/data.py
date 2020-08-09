import os
import sys
sys.path.append("..")
import core.config as config
import cv2
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

class PnetImageLoader(data.Dataset):
    def __init__(self, annotation_file, transform, mode="train"):
        self.mode = mode
        with open(annotation_file, 'r') as f:
            self.annotation = f.readlines()
        self.transform = transform

    def __getitem__(self, index):
        anno = self.annotation[index].split(' ')
        image = Image.open(anno[0])
        image = self.transform(image)
        label = torch.Tensor(np.array([int(anno[1])])).float()
        bbox = torch.Tensor(np.array(list(map(float, anno[2:6])))).float()
        landmark = torch.Tensor(np.array(list(map(float, anno[6:])))).float()
        return image, label, bbox, landmark

    def __len__(self):
        return len(self.annotation)
