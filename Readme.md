# Glaucoma detection with segmentation

### Introduction

This work is a re-implementation of previous work [Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image, 2018](https://arxiv.org/pdf/1805.07549.pdf). Project [website](https://hzfu.github.io/proj_glaucoma_fundus.html) shows the development team's work. If you have a brief glance at the source code shown on the Github, you may find a minor difference on the U-Net architecture. This is because the author wish to integrate his another work [MNet](https://arxiv.org/abs/1801.00926) into this paper. The major optimization considers decoders of all levels in U-Net as valid segmentation outputs with equal weight.

### Dataset

Dataset comes from the newer version of Refuge Challenge -- [Refuge2 2020](https://refuge.grand-challenge.org/Home2020/).  Either the older version or the newer version possesses a full set of medical eye images. Sample raw image and segmented image are shown in the sample folder.

### Code Info

Codes were reproduced with no tricks in the naive version. The starting point lies in `StreamEnsemble.py`, where you can have a complete network training to obtain four trained ResNet50 like network. Training dataset was transformed into a list called `imgList.txt` in the repository. With well-trained models, you can have you new data predicted in the FullPredict function.

I set `batch_size = 1` in segmentation training on purpose. Be careful when you intend to set it to another value because I used `torch.squeeze`.

### Paper Review

Undoubtedly this is an excellent paper filled with avant-guard ideas, at least by the time of publication. But truth to be told, personally, this paper is no longer useful since it has so many restrictions. First of all, the origin ORIGA dataset that the network was pretrained on is a small one, therefore the author resized the images to 640 * 640. With the flourishment of graphics industry, this network is incompatible with big and high-resolution images. Secondly, the network is a bit clumsy with four classification branches, though the prediction speed is 2 FPS on a single commercial GPU. I believe the state-of-the-art works could easily prevent this issue.