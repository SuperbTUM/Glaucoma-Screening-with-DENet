# Glaucoma detection with segmentation

### Introduction

This work is a combination of previous works [Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image, 2018](https://arxiv.org/pdf/1805.07549.pdf), [Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation, 2018](https://arxiv.org/abs/1801.00926), [Detection of Pathological Myopia and Optic Disc Segmentation with Deep Convolutional Neural Networks, 2019](https://ieeexplore.ieee.org/document/8929252) and [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507). Project [website](https://hzfu.github.io/proj_glaucoma_fundus.html) shows the development team's work for the first two papers. The third and fourth paper help to improve the segmentation network.

### Dataset

Dataset comes from the newer version of Refuge Challenge -- [Refuge2 2020](https://refuge.grand-challenge.org/Home2020/).  Either the older version or the newer version possesses a full set of medical eye images. Sample raw image and segmented image are shown in the sample folder. The ORIGA dataset may no longer available for download because I am unable to do that at present.

### Code Info

Codes were reproduced with no tricks in the naive version. The starting point lies in `StreamEnsemble.py`, where you can have a complete network training to obtain four trained ResNet50 like network. Training dataset was transformed into a list called `imgList.txt` in the repository. With well-trained models, you can have you new data predicted in the FullPredict function.

I set `batch_size = 1` in segmentation training on purpose. Be careful when you intend to set it to another value because I used `torch.squeeze`.

### Implementation Details

The biggest drawback is the immutable image size in the segmentation network. The modified segmentation network can be viewed as a mixture of UNet, residual block and squeeze-and-excitation block. Inspired by [Patch-Based Output Space Adversarial Learning
for Joint Optic Disc and Cup Segmentation](https://arxiv.org/abs/1902.07519), we can enrich the loss function by adding smooth loss. The classification network is ResNet50. I have not figured out a way of improving classification network. An updated version will be released if any significant progress is made.