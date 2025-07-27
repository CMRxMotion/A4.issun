## Introduction
This is the official code submitted to the CMRxMotion challenge by Team issun. 
It's also the Final Dockerfile submitted to MICCAI 2022 [CMRxMotion Challenge](http://cmr.miccai.cloud/)

<!-- arxiv Version with the GitHub code link in the paper:
[]()

Springer Version without GitHub code link in the paper:
[]() -->


## Datasets
CMRxMotion Dataset: [http://cmr.miccai.cloud/data/](http://cmr.miccai.cloud/data/)

## Methodology and Poster Overview
<!-- ![Poster](poster.png) -->
<img src="./poster.png" alt="Poster" width="1600">

## Usage
This repository has been made publicly available with the consent of Team issun under the Apache 2.0 License.

## Citation
If this code is useful for your research, please consider citing:

```
@inproceedings{10.1007/978-3-031-23443-9_45,
author = {Sun, Xiaowu and Cheng, Li-Hsin and van der Geest, Rob J.},
title = {Combination Special Data Augmentation and Sampling Inspection Network for Cardiac Magnetic Resonance Imaging Quality Classification},
year = {2022},
isbn = {978-3-031-23442-2},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-23443-9_45},
doi = {10.1007/978-3-031-23443-9_45},
abstract = {Cardiac magnetic resonance imaging (MRI) may suffer from motion-related artifacts resulting in non-diagnostic quality images. Therefore, image quality assessment (IQA) is essential for the cardiac MRI analysis. The CMRxMotion challenge aims to develop automatic methods for IQA. In this paper, given the limited amount of training data, we designed three special data augmentation techniques to enlarge the dataset and to balance the class ratio. The generated dataset was used to pre-train the model. We then randomly selected two multi-channel 2D images from one 3D volume to mimic sample inspection and introduced ResNet as the backbone to extract features from those two 2D images. Meanwhile, a channel-based attention module was used to fuse the features for the classification. Our method achieved a mean accuracy of 0.75 and 0.725 in 4-fold cross validation and the held-out validation dataset, respectively. The code can be found here ().},
booktitle = {Statistical Atlases and Computational Models of the Heart. Regular and CMRxMotion Challenge Papers: 13th International Workshop, STACOM 2022, Held in Conjunction with MICCAI 2022, Singapore, September 18, 2022, Revised Selected Papers},
pages = {476–484},
numpages = {9},
keywords = {Data augmentation, Cardiac MRI, Image quality assessment},
location = {Singapore, Singapore}
}
```
