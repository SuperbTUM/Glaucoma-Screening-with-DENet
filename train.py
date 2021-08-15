import torch
from core_train import core_train
from fc_classification import FCNet
from torch import optim
from dataset import Refuge2, Resize2_640
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
from dedicated_Resnet50 import ResNet50_Mod
import numpy as np
import cv2


def data_preload(data_paths, gt_labels, gt_segmentations, transform=None, isArray=True, isTrain=True):
    if isArray:
        dataset = Refuge2(data_paths, gt_labels, gt_segmentations, transform, isTrain)
    else:
        data = []
        for path in data_paths:
            data.append(cv2.imread(path.strip('\n')))
        data = np.array(data)
        dataset = Refuge2(data, gt_labels, gt_segmentations, transform, isTrain)
    return dataset


def getModel(base_lr=1e-3, cuda=False):
    model = FCNet()
    if cuda:
        model = model.cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30)
    return model, optimizer, lr_scheduler


def getResNet(base_lr=1e-3, cuda=False):
    model = ResNet50_Mod()
    if cuda:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30)
    return model, optimizer, lr_scheduler


def test(data, model, cuda):
    dataloader = DataLoader(data, shuffle=False, num_workers=1)
    iterator = tqdm(dataloader)
    TP = FP = TN = FN = 0.
    for sample in iterator:
        img, gt_label, gt_segmentation = sample
        if cuda:
            img = Variable(img).cuda
        classification = model(img)
        for i in range(classification.shape[0]):
            if gt_label[i] == 1:
                if classification[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if classification[i] == 1:
                    FP += 1
                else:
                    FN += 1
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    BAcc = (Sen + Spe) / 2
    return BAcc


def train_fcnet(data_paths, gt_labels, gt_segmentations, cuda):
    res = core_train(data_paths, gt_labels, gt_segmentations)
    model, optimizer, lr_scheduler = getModel(cuda=cuda)
    transform = Compose(
        [Resize2_640()]
    )
    dataset = data_preload(data_paths, gt_labels, gt_segmentations, transform=transform, isArray=False)
    epoch = 0
    best_BAcc = 0.
    while True:
        if epoch > 0 and epoch % 2 == 0:
            model.eval()
            BAcc = test(dataset, model, cuda)
            best_BAcc = max(best_BAcc, BAcc)
            model.train()
        if epoch >= 10:
            break
        dataloader = DataLoader(dataset, shuffle=True, num_workers=1)
        iterator = tqdm(dataloader)
        for sample in iterator:
            optimizer.zero_grad()
            img, gt_label, gt_segmentation = sample
            if cuda:
                img = Variable(img).cuda()
                classification = model(img).cpu()
            else:
                classification = model(img)
            loss = nn.BCELoss()(classification.view(classification.shape[0], -1), gt_label.view(gt_label.shape[0], -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
            epoch += 1
    return res, model


def train_resnet(data_paths, gt_labels, gt_segmentations, cuda, transform=None, isArray=True):
    model, optimizer, lr_scheduler = getResNet(cuda=cuda)
    dataset = data_preload(data_paths, gt_labels, gt_segmentations, transform, isArray)
    epoch = 0
    best_BAcc = 0.
    while True:
        if epoch > 0 and epoch % 2 == 0:
            model.eval()
            BAcc = test(dataset, model, cuda)
            best_BAcc = max(best_BAcc, BAcc)
            model.train()
        if epoch >= 10:
            break
        dataloader = DataLoader(dataset, shuffle=True, num_workers=1)
        iterator = tqdm(dataloader)
        for sample in iterator:
            optimizer.zero_grad()
            img, gt_label, gt_segmentation = sample
            if cuda:
                img = Variable(img).cuda()
                classification = model(img).cpu()
            else:
                classification = model(img)
            loss = nn.BCELoss()(classification.view(classification.shape[0], -1), gt_label.view(gt_label.shape[0], -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
            epoch += 1
    return model


if __name__ == "__main__":
    a = torch.ones((10, 1))
    if a[0] == 1.:
        print('True')
