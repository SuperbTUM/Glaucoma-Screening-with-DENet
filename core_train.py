import torch
from UNet import UNet
from pathlib import Path
from torch import optim
from dataset import Refuge2, Resize2_640
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
from utils import DiceLoss
from torchvision.transforms import Compose
from utils import counting_correct


def load_model(base_lr=1e-3, pretrained=None, cuda=False):
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
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
    return model, optimizer, lr_scheduler


def test(data, model, cuda):
    dataloader = DataLoader(data, shuffle=False, num_workers=1)
    iterator = tqdm(dataloader)
    correct = 0
    res = []
    for sample in iterator:
        img, gt_label, gt_segmentation = sample
        if cuda:
            img = Variable(img).cuda
        localization = model(img)
        res.append(localization.squeeze().detach().numpy())
        correct += counting_correct(localization, gt_segmentation)
    return correct / (640**2), res


def core_train(data_paths, gt_labels, gt_segmentations, batch_size=1, cuda=False):
    # data_paths = [i for i in range(400)]
    # gt_labels = [1 for _ in range(40)] + [0 for _ in range(360)]
    # gt_segmentations = []
    transform = Compose(
        [Resize2_640()]
    )
    epoch = 0
    best_PA = 0.
    model, optimizer, lr_scheduler = load_model(cuda=cuda)
    train_dataset = Refuge2(data_path=data_paths, labels=gt_labels, segmentations=gt_segmentations,
                            transform=transform)
    while True:
        if epoch > 0 and epoch % 2 == 0:
            model.eval()
            PA, res = test(train_dataset, model, cuda=cuda)
            best_PA = max(best_PA, PA)
            print('Best PA: {}'.format(best_PA))
            model.train()
        if epoch >= 10:
            break
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        iterator = tqdm(dataloader)
        for sample in iterator:
            img, gt_label, gt_segmentation = sample
            optimizer.zero_grad()
            if cuda:
                img = Variable(img).cuda()
                localization = model(img).cpu
            else:
                localization = model(img)
            loss_segmentation = DiceLoss(localization.view(localization.shape[0], -1), gt_segmentation.view(
                gt_segmentation.shape[0], -1))
            loss_segmentation.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
            epoch += 1
    return res
