# Glaucoma detection with segmentation

### Introduction

This work is a re-implementation of previous work [Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image, 2018](https://arxiv.org/pdf/1805.07549.pdf). Project [website](https://hzfu.github.io/proj_glaucoma_fundus.html) shows the development team's work. If you have a brief glance at the source code shown on the Github, you may find a minor difference on the U-Net architecture. This is because the author wish to integrate his another work [MNet](https://arxiv.org/abs/1801.00926) into this paper. The major optimization considers decoders of all levels in U-Net as valid segmentation outputs with equal weight.

### Dataset

Dataset comes from the newer version of Refuge Challenge -- [Refuge2 2020](https://refuge.grand-challenge.org/Home2020/).  The truth is no segmentation ground truth exists in dataset. I wonder how the network works.

### Code Info

Codes were reproduced with no tricks in the naive version. The starting point lies in `StreamEnsemble.py`, where you can have a complete network training to obtain four trained ResNet50 like network. Training dataset was transformed into a list called `imgList.txt` in the repository. With well-trained models, you can have you new data predicted in the FullPredict function.