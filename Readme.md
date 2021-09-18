# Glaucoma detection with segmentation

### Introduction

This work is initially a combination of previous works [Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image, 2018](https://arxiv.org/pdf/1805.07549.pdf), [Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation, 2018](https://arxiv.org/abs/1801.00926), [Detection of Pathological Myopia and Optic Disc Segmentation with Deep Convolutional Neural Networks, 2019](https://ieeexplore.ieee.org/document/8929252) and [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507). Project [website](https://hzfu.github.io/proj_glaucoma_fundus.html) shows the development team's work for the first two papers. The third and fourth paper help to improve the segmentation network.

With more reading, [Regression and Learning with Pixel-wise Attention for Retinal Fundus Glaucoma Segmentation and Detection](https://arxiv.org/pdf/2001.01815.pdf) provides X-UNet in segmentation and DeepLabV3+ in classification. Also, the author correct segmentation work from classification to regression.

### Dataset

Dataset comes from the newer version of Refuge Challenge -- [Refuge2 2020](https://refuge.grand-challenge.org/Home2020/).  Either the older version or the newer version possesses a full set of medical eye images. Sample raw image and segmented image are shown in the sample folder. The ORIGA dataset may no longer available for download because download entrance has been closed (maybe it is available through net disk share or cloud share) but pretrained model is available [here](https://pan.baidu.com/s/1eDT0N4tQsWI4McyGB36vLw?errmsg=Auth+Login+Params+Not+Corret&errno=2&ssnerror=0#list/path=%2F).

### Code Info

Codes were reproduced with my own understanding. There are some tricks like using separable convolution and Xception net. Separable convolution is widely used for simplicity and Xception net can be replaced by naive ResNet (DeepLab network needs a proper backbone. For ResNet, the deeper the better, at least at the first glance.). The evaluation functions in segmentation and classification are Dice Similarity Coefficient and BAcc respectively. Loss functions are Dice loss and Binary Cross Entropy loss. The starting point lies in `StreamEnsemble.py`, where you can have a complete network training to obtain four trained ResNet50-based network. Training dataset was transformed into a list called `imgList.txt` in the repository. With fully-trained models, you can have you new data predicted in the FullPredict function.

I set `batch_size = 1` in segmentation training by default.

### Implementation Details

The biggest drawback is the immutable image size in the segmentation network. The modified segmentation network can be viewed as a mixture of UNet, residual block and squeeze-and-excitation block. Inspired by [Patch-Based Output Space Adversarial Learning for Joint Optic Disc and Cup Segmentation](https://arxiv.org/abs/1902.07519), we can enrich the loss function by adding interaction with neighbor pixels. The classification network baseline is ResNet50, and for improvement, we have ResNet50 / Xception & DeepLab+3.

An updated version will be released if any significant progress is made.