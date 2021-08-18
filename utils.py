import torch
from torchvision.transforms import CenterCrop
import numpy as np
from skimage.transform import rotate
import cv2
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


def DiceLoss(predict, gt):
    cross_prod = predict * gt
    cross_prod = cross_prod.sum(dim=1)
    self_prod = (predict * predict).sum(dim=1) + (gt * gt).sum(dim=1)
    return 1 - 2 * cross_prod / self_prod


def counting_correct(predict, gt, threshold=0.5):
    predict = predict.view(predict.shape[0], -1)
    gt = gt.view(gt.shape[0], -1)
    is_equal = torch.eq(predict > threshold, gt < 200)  # if the background is white
    is_equal = is_equal.detach().numpy()
    return np.count_nonzero(is_equal)


def RegionCrop(origin_img, localization, threshold=0.5, size=(224, 224)):
    H, W = localization.shape
    left = W
    right = 0
    top = H
    bottom = 0
    for row in range(H):
        for col in range(W):
            if localization[row, col] > threshold:
                left = min(left, col)
                right = max(right, col)
                top = min(top, row)
                bottom = max(bottom, row)
    if left > right or top > bottom:
        raise ValueError("No disc found!\n")
    return CenterCrop(size=size)(origin_img[:, top:bottom, left:right])


# This function is of no use.
def polarTransformation(radius, theta, phai=0, size=224):
    u0 = v0 = size // 2
    u = u0 + radius * torch.cos(theta + phai)
    v = v0 + radius * torch.sin(theta + phai)
    return u, v


# This function is of no use.
def inversePolarTransformation(u, v, phai=0, size=224):
    u0 = v0 = size // 2
    radius = torch.sqrt((u - u0) * (u - u0) - (v - v0) * (v - v0))
    theta = torch.tanh((u - u0) / (v - v0)) - phai
    return radius, theta


def transformation(cropped_imgs):
    polar_imgs = []
    for img in cropped_imgs:
        img = img.transpose(1, 2, 0)
        rotate_img = rotate(cv2.linearPolar(img, (img.shape[1], img.shape[0]),
                                            img.shape[0], cv2.WARP_FILL_OUTLIERS), -90)
        polar_imgs.append(rotate_img.transpose(2, 0, 1))
    polar_imgs = np.array(polar_imgs)
    return polar_imgs


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def collate_fn(batch):
    imgs = []
    gts = []
    for sample in batch:
        if len(sample) > 1:
            img, gt = sample
            imgs.append(img)
            gts.append(gt)
        else:
            imgs.append(sample)
    imgs = torch.stack(imgs)
    if gts:
        if isinstance(gts[0], int):
            gts = torch.Tensor(gts).int()
        else:
            gts = torch.stack(gts)
        return imgs, gts
    else:
        return imgs


if __name__ == "__main__":
    a = [[3, 1], [2, 4]]
    a = torch.Tensor(a)
    a.detach().numpy()
