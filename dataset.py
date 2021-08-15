import cv2
from torch.utils.data import Dataset
import torch


class Resize2_640(object):
    def __init__(self, size=(640, 640)):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, self.size)


class Refuge2(Dataset):
    def __init__(self, data, labels, segmentations=None, transform=None, isTrain=True):
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
            label = self.labels[item]
            if self.segmentations:
                seg_img = cv2.imread(self.segmentations[item])
            else:
                seg_img = None
            if self.transform:
                img = self.transform(img)
                seg_img = torch.Tensor(self.transform(seg_img)) if self.segmentations else None
            return torch.Tensor(img), torch.Tensor(label), seg_img
        else:
            if self.transform:
                img = self.transform(img)
            return torch.Tensor(img)


if __name__ == "__main__":
    with open('imgList.txt', 'r') as f:
        imgs = f.readlines()
    imgs = list(map(lambda x:x.strip('\n'), imgs))
    f.close()
    cv2.imread(imgs[0])
