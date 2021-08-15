import torch
from train import train_fcnet, train_resnet
from utils import RegionCrop, transformation
from train import data_preload
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable


def FullTrain(imgs, data_paths, gt_labels, gt_segmentations, cuda):
    # global image
    model_global = train_resnet(data_paths, gt_labels, None, cuda=cuda, transform=None)
    # Cropped image
    discs, model_fc = train_fcnet(data_paths, gt_labels, gt_segmentations, cuda)
    cropped_imgs = []
    for i in range(len(discs)):
        cropped_imgs.append(RegionCrop(imgs[i], discs[i]))
    cropped_imgs = torch.stack(cropped_imgs)
    model_cropped = train_resnet(cropped_imgs, gt_labels, None, cuda, transform=None)
    # transformation
    polar_imgs = transformation(cropped_imgs)
    model_polar = train_resnet(polar_imgs, gt_labels, None, cuda, transform=None)
    return model_global, model_fc, model_cropped, model_polar


def FullPredict(imgs, models, cuda=False):
    model_global, model_fc, model_cropped, model_polar = models
    imgs = data_preload(imgs, None, None, transform=None, isTrain=False)
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
        predict = (predict_global + predict_fc + predict_cropped + predict_polar) / 4
        predicts.append(predict)
    return predicts
