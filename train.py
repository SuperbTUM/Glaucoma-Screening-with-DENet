import torch
from core_train import core_train
from fc_classification import FCNet
from torch import optim
from dataset import Refuge2, Resize2_640, RandomRotation, RandomFlip
from torchvision.transforms import Compose
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
from dedicated_Resnet50 import ResNet50_Mod
from utils import DataLoaderX, collate_fn
from sklearn.metrics import roc_auc_score
import numpy as np


def getModel(base_lr=1e-4, cuda=False):
    model = FCNet()
    if cuda:
        model = model.cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30)
    return model, optimizer, lr_scheduler


def getResNet(size, base_lr=1e-4, cuda=False):
    model = ResNet50_Mod(input_size=size)
    if cuda:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30)
    return model, optimizer, lr_scheduler


def test(data, model, batch_size, cuda):
    dataloader = DataLoaderX(data, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)
    iterator = tqdm(dataloader)
    classification = list()
    gts = list()
    for sample in iterator:
        img, gt_label = sample
        if cuda:
            img = Variable(img).cuda
        classification.append(model(img).cpu().detach().numpy())
        gts.append(gt_label.numpy())
    classification = np.stack(classification).flatten()
    gts = np.stack(gts).flatten()
    auc = roc_auc_score(classification, gts, average=None)
    return auc


def train_fcnet(data, gt_labels, gt_segmentations, batch_size=1, cuda=False):
    res = core_train(data, None, gt_segmentations, cuda=cuda)
    model, optimizer, lr_scheduler = getModel(cuda=cuda)
    transform = Compose(
        [
            RandomRotation(),
            RandomFlip(),
            Resize2_640()
        ]
    )
    dataset = Refuge2(data, gt_labels, None, transform=transform)
    epoch = 0
    best_auc = 0.
    while True:
        if epoch > 0 and epoch % 2 == 0:
            model.eval()
            auc = test(dataset, model, batch_size, cuda)
            best_auc = max(best_auc, auc)
            model.train()
        if epoch >= 10:
            break
        dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
        iterator = tqdm(dataloader)
        for sample in iterator:
            optimizer.zero_grad()
            img, gt_label = sample
            if cuda:
                img = Variable(img).cuda()
                classification = model(img).cpu()
            else:
                classification = model(img)
            # loss = nn.BCELoss()(classification.view(classification.shape[0], -1), gt_label.view(gt_label.shape[0], -1))
            loss = nn.L1Loss()(classification.view(classification.shape[0], -1), gt_label.view(gt_label.shape[0], -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
            epoch += 1
    return res, model


def train_resnet(data, gt_labels, batch_size=1, cuda=False):
    print("**********************************Start training resnet*********************************")
    model, optimizer, lr_scheduler = getResNet(size=data.shape[1], cuda=cuda)
    transform = Compose(
        [
            RandomRotation(),
            RandomFlip()
        ]
    )
    dataset = Refuge2(data, gt_labels, segmentations=None, transform=transform)
    epoch = 0
    best_auc = 0.
    while True:
        if epoch > 0 and epoch % 2 == 0:
            model.eval()
            auc = test(dataset, model, batch_size=batch_size, cuda=cuda)
            best_auc = max(best_auc, auc)
            model.train()
        if epoch >= 10:
            break
        dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
        iterator = tqdm(dataloader)
        for sample in iterator:
            optimizer.zero_grad()
            img, gt_label = sample
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
    a = [0 for _ in range(10)]
    a = torch.Tensor(a)
    a = a.numpy()
    b = [a, a]
    b = np.stack(b).flatten()
    print(b.shape)
