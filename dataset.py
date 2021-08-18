import cv2
from torch.utils.data import Dataset
import torch
import numpy as np


class Resize2_640(object):
    def __init__(self, size=(640, 640)):
        self.size = size

    def __call__(self, imgs):
        img_list = []
        sample_img = imgs[0]
        scale = max(sample_img.shape[0], sample_img.shape[1]) / self.size[0]
        adjust_size = (sample_img[0] // scale, sample_img[1] // scale)
        if scale > 1:  # zoom out
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
            print('Not implemented!\n')
        for img in imgs:
            img = img.transpose(1, 2, 0)
            img_with_ratio = cv2.resize(img, adjust_size, interpolation=interpolation)
            top = (self.size[0] - adjust_size[0]) // 2
            bottom = self.size[0] - adjust_size[0] - top
            left = (self.size[1] - adjust_size[1]) // 2
            right = self.size[1] - adjust_size[1] - left
            resized_image_with_ratio = cv2.copyMakeBorder(img_with_ratio, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                          [0, 0, 0])
            img_list.append(resized_image_with_ratio.transpose(2, 0, 1))
        return img_list


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, imgs):
        angles = [0, 90, 180, 270]
        prob = np.random.randint(0, len(angles))
        img_list = []
        for img in imgs:
            img = img.transpose(1, 2, 0)
            M = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), angles[prob], 1)
            rotated = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))
            img_list.append(rotated.transpose(2, 0, 1))
        return img_list


class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, imgs):
        if np.random.random(1) > self.prob:
            return imgs
        img_list = []
        for img in imgs:
            img = img.transpose(1, 2, 0)
            img_list.append(cv2.flip(img, -1).transpose(2, 0, 1))
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
                label = self.labels[item]
                if self.transform:
                    img = self.transform([img])[0]
                return torch.Tensor(img), label
            if self.segmentations:
                seg_img = self.segmentations[item]
                if self.transform:
                    img, seg_img = self.transform((img, seg_img))
                return torch.Tensor(img), torch.Tensor(seg_img)
        else:
            if self.transform:
                img = self.transform(img)
            return torch.Tensor(img)


if __name__ == '__main__':
    a = [torch.Tensor([1]) for _ in range(10)]
    print(a)
