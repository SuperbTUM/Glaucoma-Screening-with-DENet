import torch
from torchvision.transforms import CenterCrop
import numpy as np
from skimage.transform import rotate
import cv2
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


def smoothLoss(predict, gt, reduction='mean'):
    H, W = predict.shape[2:]
    loss = torch.abs(predict[:, :, 1:H - 1, 1:W - 1] - predict[:, :, 0:H - 2, 1:W - 1]) + \
           torch.abs(predict[:, :, 1:H - 1, 1:W - 1] - predict[:, :, 2:H, 1:W - 1]) + \
           torch.abs(predict[:, :, 1:H - 1, 1:W - 1] - predict[:, :, 1:H - 1, 0:W - 2]) + \
           torch.abs(predict[:, :, 1:H - 1, 1:W - 1] - predict[:, :, 1:H - 1, 2:W])
    M1 = torch.eq(gt[:, :, 1:H-1, 1:W-1], gt[:, :, 0:H-2, 1:W-1]).float()
    M2 = torch.eq(gt[:, :, 1:H-1, 1:W-1], gt[:, :, 2:H, 1:W-1]).float()
    M3 = torch.eq(gt[:, :, 1:H-1, 1:W-1], gt[:, :, 1:H-1, 0:W-2]).float()
    M4 = torch.eq(gt[:, :, 1:H-1, 1:W-1], gt[:, :, 1:H-1, 2:W]).float()
    mask = M1 * M2 * M3 * M4
    if reduction == 'mean':
        return (loss * mask).mean()
    else:
        return (loss * mask).sum()


def DiceLoss(predict, gt, thresholdG=200, epsilon=1e-8, reduction='mean'):
    predict = predict.view(predict.shape[0], -1)
    gt = gt.view(gt.shape[0], -1)
    gt = torch.where(gt < thresholdG, 1, 0)
    cross_prod = predict * gt
    cross_prod = cross_prod.sum(dim=-1)
    self_prod = (predict * predict).sum(dim=-1) + (gt * gt).sum(dim=-1) + epsilon
    if reduction == 'mean':
        return (1 - 2 * cross_prod / self_prod).mean()
    elif reduction == 'sum':
        return (1 - 2 * cross_prod / self_prod).sum()
    else:
        raise NotImplementedError


def DiceSmoothLoss(predict, gt, lambda1=0.4, lambda2=1., thresholdG=200, epsilon=1e-8, reduction='mean'):
    return lambda1 * DiceLoss(predict, gt, thresholdG, epsilon, reduction) + \
           lambda2 * smoothLoss(predict, gt)


def DSC(predict, gt, thresholdP=0.5, thresholdG=200):  # If the background is white
    predict = predict.flatten()
    predict = torch.where(predict > thresholdP, 1, 0)
    gt = gt.flatten()
    gt = torch.where(gt < thresholdG, 1, 0)
    TP = FN = FP = 0
    pairs = list(zip(gt, predict))
    for gt_pixel, pred_pixel in pairs:
        if gt_pixel and pred_pixel:
            TP += 1
        elif (gt_pixel or pred_pixel) == 0:
            FN += 1
        elif gt_pixel == 0 and pred_pixel == 1:
            FP += 1
    return 2 * TP / (2 * TP + FN + FP)


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
    a = torch.ones((10, 1, 10, 10))
    b = torch.ones((10, 1, 10, 10))
    print(DiceSmoothLoss(a, b))
