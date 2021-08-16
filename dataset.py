import cv2
from torch.utils.data import Dataset
import torch
import numpy as np


class Resize2_640(object):
    def __init__(self, size=(640, 640)):
        self.size = size

    def __call__(self, imgs):
        img_list = []
        for img in imgs:
            if img is None:
                continue
            img_list.append(cv2.resize(img, self.size, interpolation=cv2.INTER_AREA))
        return img_list


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, imgs):
        angles = [0, 90, 180, 270]
        prob = np.random.randint(0, len(angles))
        img_list = []
        for img in imgs:
            if img is None:
                continue
            M = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), angles[prob], 1)
            rotated = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))
            img_list.append(rotated)
        return img_list


class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, imgs):
        if np.random.random(1) > self.prob:
            return imgs
        img_list = []
        for img in imgs:
            if img is None:
                continue
            img_list.append(cv2.flip(img, -1))
        return img_list


class Refuge2(Dataset):
    def __init__(self, data, labels=None, segmentations=None, transform=None, isTrain=True):
        super(Refuge2, self).__init__()
        self.data = data
        self.transform = transform
        self.isTrain = isTrain
        if self.isTrain:
            self.labels = labels
            self.segmentations = segmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        if self.isTrain:
            if self.labels:
                label = torch.Tensor(self.labels[item])
            else:
                label = None
            if self.segmentations:
                seg_img = cv2.imread(self.segmentations[item].strip('\n'))
            else:
                seg_img = None
            if self.transform:
                img_list = self.transform((img, seg_img))
                img = img_list[0]
                if len(img_list) > 1:
                    seg_img = torch.Tensor(img_list[1])
            seg_img = torch.Tensor(seg_img) if seg_img is not None else None
            return torch.Tensor(img), label, seg_img
        else:
            if self.transform:
                img = self.transform(img)
            return torch.Tensor(img)


