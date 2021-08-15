import torch
from torchvision.transforms import CenterCrop
import numpy as np
from skimage.transform import rotate
import cv2


def DiceLoss(predict, gt):
    cross_prod = predict * gt
    cross_prod = cross_prod.sum(dim=1)
    self_prod = (predict * predict).sum(dim=1) + (gt * gt).sum(dim=1)
    return 1 - cross_prod / self_prod


def counting_correct(predict, gt):
    predict = predict.view(predict.shape[0], -1)
    gt = gt.view(gt.shape[0], -1)
    is_equal = torch.eq(predict, gt)
    is_equal = is_equal.detach().numpy()
    return np.count_nonzero(is_equal)


def RegionCrop(origin_img, localization, size=(224, 224)):
    C, H, W = localization.shape
    left = W
    right = 0
    top = H
    bottom = 0

    has_target = localization.any(dim=0)
    for row in range(W):
        for col in range(H):
            if has_target[row, col]:
                left = min(left, col)
                right = max(right, col)
                top = min(top, row)
                bottom = max(bottom, row)
    if left > right or top > bottom:
        raise ValueError("No disc found!\n")
    return CenterCrop(size=size)(origin_img[:, top:bottom, left:right])


def polarTransformation(radius, theta, phai=0, size=224):
    u0 = v0 = size // 2
    u = u0 + radius * torch.cos(theta+phai)
    v = v0 + radius * torch.sin(theta+phai)
    return u, v


def inversePolarTransformation(u, v, phai=0, size=224):
    u0 = v0 = size // 2
    radius = torch.sqrt((u-u0)*(u-u0) - (v-v0)*(v-v0))
    theta = torch.tanh((u-u0)/(v-v0)) - phai
    return radius, theta


def transformation(cropped_imgs):
    polar_imgs = []
    for img in cropped_imgs:
        polar_imgs.append(rotate(cv2.linearPolar(cropped_imgs, (400, 400), 400, cv2.WARP_FILL_OUTLIERS), -90))
    polar_imgs = np.array(polar_imgs)
    return polar_imgs


if __name__ == "__main__":
    a = [[True, False], [False, False]]
    a = torch.Tensor(a)
    a = a.numpy()
    print(np.count_nonzero(a))