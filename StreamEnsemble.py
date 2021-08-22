import torch
from train import train_fcnet, train_resnet
from utils import RegionCrop, transformation
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import cv2
from dataset import Refuge2, Resize2_640
from torchvision.transforms import Compose
import numpy as np


def load_train_images():
    imgs_array = []
    with open('imgList.txt', 'r') as f:
        names = f.readlines()
        for name in names:
            name = name.strip('\n')
            imgs_array.append(cv2.imread(name).transpose(2, 0, 1))
    f.close()
    return np.array(imgs_array)


def load_gt_labels(path='imgList.txt'):
    if path == 'imgList.txt':
        labels = [1 for _ in range(40)] + [0 for _ in range(360)]
        return tuple(labels)
    else:
        raise NotImplementedError


def load_segment_images():
    imgs_array = []
    with open('segList.txt', 'r') as f:
        names = f.readlines()
        for name in names:
            name = name.strip('\n')
            imgs_array.append(cv2.imread(name, 0))
    f.close()
    return np.array(imgs_array)


def load_predict_imgs(path):
    imgs_array = []
    with open(path, 'r') as f:
        names = f.readlines()
        for name in names:
            name = name.strip('\n')
            imgs_array.append(cv2.imread(name).transpose(2, 0, 1))
    f.close()
    return np.array(imgs_array)


def FullTrain(imgs, gt_labels, gt_segmentations, cuda=False):
    # global image
    model_global = train_resnet(imgs, gt_labels, batch_size=10, cuda=cuda)
    # Cropped image
    discs, model_fc = train_fcnet(imgs, gt_labels, gt_segmentations, cuda)
    cropped_imgs = []
    for i in range(len(discs)):
        cropped_imgs.append(RegionCrop(imgs[i], discs[i]))
    cropped_imgs = np.stack(cropped_imgs)
    model_cropped = train_resnet(cropped_imgs, gt_labels, batch_size=10, cuda=cuda)
    # transformation
    polar_imgs = transformation(cropped_imgs)
    model_polar = train_resnet(polar_imgs, gt_labels, batch_size=10, cuda=cuda)
    return model_global, model_fc, model_cropped, model_polar


def FullPredict(imgs, models, cuda=False):
    model_global, model_fc, model_cropped, model_polar = models
    transform = Compose(
        [Resize2_640()]
    )
    imgs = Refuge2(imgs, None, None, transform=transform, isTrain=False)
    dataloader = DataLoader(imgs, shuffle=False, num_workers=1)
    iterator = tqdm(dataloader)
    predicts = list()
    for img in iterator:
        if cuda:
            img = Variable(img).cuda()
        predict_global = model_global(img)
        predict_fc = model_fc(img)
        predict_cropped = model_cropped(img)
        predict_polar = model_polar(img)
        predict = (predict_global + predict_fc + predict_cropped + predict_polar) // 4
        predicts.append(predict)
    return predicts


def main(predict_path, cuda=False):
    train_imgs = load_train_images()
    gt_labels = load_gt_labels()
    gt_segmentations = load_segment_images()
    model1, model2, model3, model4 = FullTrain(train_imgs, gt_labels, gt_segmentations, cuda)
    if predict_path:
        predict_imgs = load_predict_imgs(predict_path)
        predicts = FullPredict(predict_imgs, (model1, model2, model3, model4), cuda)
        return predicts
    else:
        torch.save(model1.state_dict(), 'global_model.pth')
        torch.save(model2.state_dict(), 'fc_model.pth')
        torch.save(model3.state_dict(), 'cropped_model.pth')
        torch.save(model4.state.dict(), 'polar_model.pth')
        return


if __name__ == "__main__":
    main(None)
