import torch
from UNet import UNet
from pathlib import Path
from torch import optim
from dataset import Refuge2, Resize2_640, RandomRotation, RandomFlip
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
from utils import DiceLoss
from torchvision.transforms import Compose
from utils import DSC, DataLoaderX, collate_fn


def load_model(base_lr=1e-4, pretrained=None, cuda=False):
    model = UNet()
    if cuda:
        model = model.cuda()
    if pretrained:
        pretrained = Path(pretrained)
        with pretrained.open() as f:
            if cuda is False:
                states = torch.load(f, map_location=torch.device("cpu"))
            else:
                states = torch.load(f)
            model.load_state_dict(states)
            model.eval()
        f.close()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
    return model, optimizer, lr_scheduler


def test(data, model, batch_size, cuda):
    dataloader = DataLoaderX(data, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)
    iterator = tqdm(dataloader)
    dsc_list = list()
    res = []
    for sample in iterator:
        img, gt_segmentation = sample
        if cuda:
            img = Variable(img).cuda
        localization = model(img)
        res.append(localization)
        dsc_list.append(DSC(localization, gt_segmentation))
    return sum(dsc_list) / len(dsc_list), res


def core_train(data, gt_segmentations, batch_size=1, cuda=False):
    print("*******************************Start training disc segmentation******************************")
    transform = Compose(
        [
            RandomRotation(),
            RandomFlip(),
            Resize2_640()
        ]
    )
    epoch = 0
    best_dsc = 0.
    model, optimizer, lr_scheduler = load_model(cuda=cuda)
    train_dataset = Refuge2(data=data, labels=None, segmentations=gt_segmentations,
                            transform=transform)
    res = None
    print("*********************************Data loading completed*******************************")
    while True:
        if epoch > 0 and epoch % 2 == 0:
            model.eval()
            dsc, res = test(train_dataset, model, batch_size=batch_size, cuda=cuda)
            best_dsc = max(best_dsc, dsc)
            print('Best dsc: {}'.format(best_dsc))
            model.train()
        if epoch >= 10:
            break
        dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                 collate_fn=collate_fn)
        iterator = tqdm(dataloader)
        for sample in iterator:
            img, gt_segmentation = sample
            optimizer.zero_grad()
            if cuda:
                img = Variable(img).cuda()
                localization = model(img).cpu
            else:
                localization = model(img)
            loss_segmentation = DiceLoss(localization, gt_segmentation)
            loss_segmentation.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
            epoch += 1
    print("************************************Segmentation result obtained***********************************")
    return res
