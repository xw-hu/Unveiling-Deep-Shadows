# Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning

by Xiaowei Hu*, Zhenghao Xing*, Tianyu Wang, Chi-Wing Fu, and Pheng-Ann Heng             

https://arxiv.org/abs/2409.02108

***

This repository contains the results and trained models for deep-learning methods used in shadow detection, removal, and generation, as presented in our paper "Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning." This paper presents a comprehensive survey of shadow detection, removal, and generation in images and videos within the deep learning landscape over the past decade, covering tasks, deep models, datasets, and evaluation metrics. Key contributions include a comprehensive survey of shadow analysis, standardization of experimental comparisons, exploration of the relationships among model size, speed, and performance, a cross-dataset generalization study, identification of open issues and future directions, and provision of publicly available resources to support further research.

## Highlights
+ A Comprehensive Survey of Shadow Analysis in the Deep Learning Era.
+ Fair Comparisons of the Existing Methods. Unified platform + newly refined datasets with corrected noisy labels and ground-truth images.
+ Exploration of Model Size, Speed, and Performance Relationships. A more comprehensive comparison of different evaluation aspects.
+ Cross-Dataset Generalization Study. Assess the generalization capability of deep models across diverse datasets.
+ Overview of Open Issues and Future Directions with AIGC and Large Models.
+ Publicly Available Results, Trained Models, and Evaluation Metrics.


## Image Shadow Detection

### Comparing image shadow detection methods on SBU-Refine and CUHK-Shadow: [Results-Part1](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowDetection-1.zip), [Results-Part2](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowDetection-2.zip) 

| ![Cannot find!](./detection.png) |
|:--------------------------------------:|
|Shadow detection methods on the SBU-Refine (left) and CUHK-Shadow (right) datasets: accuracy, parameters (indicated by the area of the bubbles), and speed.|


| Input Size | Methods                                   | BER (SBU-Refine) (↓) | BER (CUHK-Shadow) (↓) | Params (M) | Infer. (images/s) |
|:----------:|:-----------------------------------------:|:----------------:|:-----------------:|:----------:|:-----------------:|
| 256x256    | DSC (CVPR 2018, TPAMI 2020)               | 6.79             | 10.97             | 122.49     | 26.86             |
|            | BDRAR (ECCV 2018)                         | 6.27             | 10.09             | 42.46      | 39.76             |
|            | DSDNet# (CVPR 2019)                       | 5.37             | 8.56              | 58.16      | 37.53             |
|            | MTMT-Net$ (CVPR 2020)                     | 6.32             | 8.90              | 44.13      | 34.04             |
|            | FDRNet (ICCV 2021)                        | 5.64             | 14.39             | 10.77      | 41.39             |
|            | FSDNet* (TIP 2021)                        | 7.16             | 9.93              | 4.39       | 150.99            |
|            | ECA (MM 2021)                             | 7.08             | 8.58              | 157.76     | 27.55             |
|            | SDDNet (MM 2023)                          | 5.39             | 8.66              | 15.02      | 36.73             |
| 512x512    | DSC (CVPR 2018, TPAMI 2020)               | 6.34             | 9.53              | 122.49     | 22.59             |
|            | BDRAR (ECCV 2018)                         | 5.62             | 8.79              | 42.46      | 31.34             |
|            | DSDNet# (CVPR 2019)                       | 5.04             | 7.79              | 58.16      | 32.69             |
|            | MTMT-Net$ (CVPR 2020)                     | 5.79             | 8.32              | 44.13      | 28.75             |
|            | FDRNet (ICCV 2021)                        | 5.39             | 6.58              | 10.77      | 35.00             |
|            | FSDNet* (TIP 2021)                        | 6.80             | 8.84              | 4.39       | 134.47            |
|            | ECA (MM 2021)                             | 7.52             | 7.99              | 157.76     | 22.41             |
|            | SDDNet (MM 2023)                          | 4.86             | 7.65              | 15.02      | 37.65             |


**Notes**:
- Evaluation on an NVIDIA GeForce RTX 4090 GPU
- $: additional training data
- *: real-time shadow detector
- #: extra supervision from other methods

### Cross-dataset generalization evaluation. Trained on SBU-Refine and tested on SRD: [Results](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowDetection_CrossVal.zip)

| Input Size | Metric | DSC (CVPR 2018, TPAMI 2020) | BDRAR (ECCV 2018) | DSDNet# (CVPR 2019) | MTMT-Net$ (CVPR 2020) | FDRNet (ICCV 2021) | FSDNet* (TIP 2021) | ECA (MM 2021) | SDDNet (MM 2023) |
|:----------:|:-------:|:------------------------:|:---------------:|:-----------------:|:-------------------:|:----------------:|:----------------:|:-----------:|:--------------:|
| 256x256    | BER (↓)   | 11.10                    | 9.13            | 10.29             | 9.97                | 11.82            | 12.13            | 11.97        | 8.64          |
| 512x512    | BER (↓)   | 11.62                    | 8.53            | 8.92              | 9.19                | 8.81             | 11.94            | 12.71        | 7.65          |


### Datasets
- [SBU-Refine](https://github.com/hanyangclarence/SILT/releases/tag/refined_sbu)
- [CUHK-Shadow](https://github.com/xw-hu/CUHK-Shadow#cuhk-shadow-dateset)
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view?pli=1), [Test](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view), and [Masks](https://yuhaoliu7456.github.io/projects/RRL-Net/index.html)

### Metric
- BER: see this GitHub repository.

## Video Shadow Detection
### Comparison of video shadow detection methods on ViSha: [Results](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/VideoShadowDetection.zip) 
| Methods                       | BER (↓) | IoU [%] (↑) | TS [%] (↑) | AVG (↑) | Param. (M) | Infer. (frames/s) |
|:-----------------------------:|:----------------:|:------------------:|:-----------------:|:--------------:|:----------:|:-----------------:|
| TVSD-Net (CVPR 2021) | 14.21            | 56.36              | 22.69             | 39.53          | 60.83      | 15.50             |
| STICT\$* (CVPR 2022)    | 13.05            | 43.75              | 39.10             | 41.43          | 26.17      | 91.34             |
| SC-Cor (ECCV 2022) | 12.80            | 55.56              | 23.68             | 39.62          | 58.16      | 27.91             |
| SCOTCH and SODA (CVPR 2023) | 10.36     | 61.24              | 25.76             | 43.50          | 53.11      | 16.16             |
| ShadowSAM (TCSVT 2023) | 13.38            | 61.72              | 23.77             | 42.75          | 93.74      | 15.53             |

**Notes**:
- Evaluation on an NVIDIA GeForce RTX 4090 GPU
- $: additional training data
- *: real-time shadow detector

### Dataset
- [ViSha](https://erasernut.github.io/ViSha.html)

### Metrics
- TS: see this GitHub repository.
- IoU: see this GitHub repository.
- BER: see this GitHub repository.

  
## Instance Shadow Detection

### Comparing image instance shadow detection methods on the SOBA-testing set: [Results](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/InstanceShadowDetection_SOBA.zip)

| Methods                        | $SOAP_{segm}$ | $SOAP_{bbox}$ | Asso. $AP_{segm}$ | Asso. $AP_{bbox}$ | Ins. $AP_{segm}$ | Ins. $AP_{bbox}$ | Param. (M) | Infer. (images/s) |
|:-------------------------------|:-------------:|:-------------:|:-----------------:|:-----------------:|:----------------:|:----------------:|:----------:|:-----------------:|
| LISA (CVPR 2020)       | 23.5          | 21.9          | 42.7              | 50.4              | 39.7             | 38.2             | 91.26      | 8.16              |
| SSIS (CVPR 2021)       | 29.9          | 26.8          | 52.3              | 59.2              | 43.5             | 41.5             | 57.87      | 5.83              |
| SSISv2 (TPAMI 2023)     | 35.3          | 29.0          | 59.2              | 63.0              | 50.2             | 44.4             | 76.77      | 5.17              |

### Comparing image instance shadow detection methods on the SOBA-challenge set: [Results](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/InstanceShadowDetection_SOBA-challenge.zip) 

| Methods                        | $SOAP_{segm}$ | $SOAP_{bbox}$ | Asso. $AP_{segm}$ | Asso. $AP_{bbox}$ | Ins. $AP_{segm}$ | Ins. $AP_{bbox}$ | Param. (M) | Infer. (images/s) |
|:-------------------------------|:-------------:|:-------------:|:-----------------:|:-----------------:|:----------------:|:----------------:|:----------:|:-----------------:|
| LISA (CVPR 2020)       | 10.2          | 9.8           | 21.6              | 26.0              | 23.9             | 24.7             | 91.26      | 4.52              |
| SSIS (CVPR 2021)       | 12.8          | 12.9          | 28.4              | 32.5              | 25.7             | 26.5             | 57.87      | 2.26              |
| SSISv2 (TPAMI 2023)     | 17.7          | 15.0          | 34.5              | 37.2              | 31.0             | 28.4             | 76.77      | 1.91              |


### Cross-dataset generalization evaluation. Trained on SOBA and tested on SOBA-VID: [Results](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/InstanceShadowDetection_SOBA-VID_CrossVal.zip)

| Methods                        | $SOAP_{segm}$ | $SOAP_{bbox}$ | Asso. $AP_{segm}$ | Asso. $AP_{bbox}$ | Ins. $AP_{segm}$ | Ins. $AP_{bbox}$ |
|:-------------------------------|:-------------:|:-------------:|:-----------------:|:-----------------:|:----------------:|:----------------:|
| LISA (CVPR 2020)               | 22.6          | 21.1          | 44.2              | 53.6              | 39.0             | 37.3             |
| SSIS (CVPR 2021)               | 32.1          | 26.6          | 58.6              | 64.0              | 46.4             | 41.0             |
| SSISv2 (TPAMI 2023)            | 37.0          | 26.7          | 63.6              | 67.5              | 51.8             | 42.8             |

### Datasets
- [SOBA](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP)
- [SOBA-VID](https://github.com/HarryHsing/Video-Instance-Shadow-Detection)

### Metrics
- SOAP: [Python](https://github.com/stevewongv/SSIS)
- SOAP-VID: [Python](https://github.com/HarryHsing/Video-Instance-Shadow-Detection)

## Image Shadow Removal

| ![Cannot find!](./removal.png) |
|:--------------------------------------:|
|Shadow removal methods on the SRD (left) and ISTD+ (right) datasets: accuracy, parameters (indicated by the area of the bubbles), and speed.|

### Comparing image shadow removal methods on SRD and ISTD+: [Results-Part1](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval-1.zip), [Results-Part2](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval-2.zip), [Results-Part3](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval-3.zip)

| Input Size | Methods                                | RMSE (SRD) (↓) | PSNR (SRD) (↑) | SSIM (SRD) (↑) | LPIPS (SRD) (↓) | RMSE (ISTD+) (↓) | PSNR (ISTD+) (↑) | SSIM (ISTD+) (↑) | LPIPS (ISTD+) (↓) | Params (M) | Infer. (images/s) |
|:----------:|:--------------------------------------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:------------:|:------------:|:-------------:|:----------:|:-----------------:|
| 256x256    | ST-CGAN (CVPR 2018)                    | 4.15      | 25.08      | 0.637      | 0.443       | 3.77        | 25.74        | 0.691        | 0.408         | 58.49      | 111.79            |
|            | SP+M-Net (ICCV 2019)                   | 5.68      | 22.25      | 0.636      | 0.444       | 3.37        | 26.58        | 0.717        | 0.373         | 54.42      | 33.88             |
|            | Mask-ShadowGAN (ICCV 2019)             | 4.32      | 24.67      | 0.662      | 0.427       | 3.70        | 25.50        | 0.720        | 0.377         | 22.76      | 64.77             |
|            | DSC (TPAMI 2020)                       | 3.97      | 25.46      | 0.678      | 0.412       | 3.44        | 26.53        | 0.738        | 0.347         | 122.49     | 57.40             |
|            | Auto (CVPR 2021)                       | 5.37      | 23.20      | 0.694      | 0.370       | 3.53        | 26.10        | 0.718        | 0.365         | 196.76     | 33.23             |
|            | G2R-ShadowNet (CVPR 2021)              | 6.08      | 21.72      | 0.619      | 0.460       | 4.37        | 24.23        | 0.696        | 0.396         | 22.76      | 3.62              |
|            | DC-ShadowNet (ICCV 2021)               | 4.27      | 24.72      | 0.670      | 0.383       | 3.89        | 25.18        | 0.693        | 0.406         | 10.59      | 40.51             |
|            | BMNet (CVPR 2022)                      | 4.39      | 24.24      | 0.721      | 0.327       | 3.34        | 26.62        | 0.731        | 0.354         | 0.58       | 17.42             |
|            | SG-ShadowNet (ECCV 2022)               | 4.60      | 24.10      | 0.636      | 0.443       | 3.32        | 26.80        | 0.717        | 0.369         | 6.17       | 16.51             |
|            | ShadowDiffusion (CVPR 2023)            | 4.84      | 23.26      | 0.684      | 0.363       | 3.44        | 26.51        | 0.688        | 0.404         | 55.52      | 9.73              |
|            | ShadowFormer (AAAI 2023)               | 4.44      | 24.28      | 0.715      | 0.348       | 3.45        | 26.55        | 0.728        | 0.350         | 11.37      | 32.57             |
|            | ShadowMaskFormer (arXiv 2024)          | 4.69      | 23.85      | 0.671      | 0.386       | 3.39        | 26.57        | 0.698        | 0.395         | 2.28       | 17.63             |
|            | HomoFormer (CVPR 2024)                 | 4.17      | 24.64      | 0.723      | 0.325       | 3.37        | 26.72        | 0.732        | 0.348         | 17.81      | 16.14             |
| 512x512    | ST-CGAN (CVPR 2018)                    | 3.44      | 26.95      | 0.786      | 0.282       | 3.36        | 27.32        | 0.829        | 0.252         | 58.49      | 52.84             |
|            | SP+M-Net (ICCV 2019)                   | 4.35      | 24.89      | 0.792      | 0.269       | 2.96        | 28.31        | 0.866        | 0.183         | 54.42      | 24.48             |
|            | Mask-ShadowGAN (ICCV 2019)             | 3.83      | 25.98      | 0.803      | 0.270       | 3.42        | 26.51        | 0.865        | 0.196         | 22.76      | 52.70             |
|            | DSC (TPAMI 2020)                       | 3.29      | 27.39      | 0.802      | 0.263       | 2.75        | 28.85        | 0.861        | 0.196         | 122.49     | 31.37             |
|            | Auto (CVPR 2021)                       | 4.71      | 24.32      | 0.800      | 0.247       | 2.99        | 28.07        | 0.853        | 0.189         | 196.76     | 28.28             |
|            | G2R-ShadowNet (CVPR 2021)              | 5.72      | 22.44      | 0.765      | 0.302       | 3.31        | 27.13        | 0.841        | 0.221         | 22.76      | 2.50              |
|            | DC-ShadowNet (ICCV 2021)               | 3.68      | 26.47      | 0.808      | 0.255       | 3.64        | 26.06        | 0.835        | 0.234         | 10.59      | 39.45             |
|            | BMNet (CVPR 2022)                      | 4.00      | 25.39      | 0.820      | 0.225       | 3.06        | 27.74        | 0.848        | 0.212         | 0.58       | 17.49             |
|            | SG-ShadowNet (ECCV 2022)               | 4.01      | 25.56      | 0.786      | 0.279       | 2.98        | 28.25        | 0.849        | 0.205         | 6.17       | 8.12              |
|            | ShadowDiffusion (CVPR 2023)            | 5.11      | 23.09      | 0.804      | 0.240       | 3.10        | 27.87        | 0.839        | 0.222         | 55.52      | 2.96              |
|            | ShadowFormer (AAAI 2023)               | 3.90      | 25.60      | 0.819      | 0.228       | 3.06        | 28.07        | 0.847        | 0.204         | 11.37      | 23.32             |
|            | ShadowMaskFormer (arXiv 2024)          | 4.15      | 25.13      | 0.798      | 0.249       | 2.95        | 28.34        | 0.849        | 0.211         | 2.28       | 14.25             |
|            | HomoFormer (CVPR 2024)                 | 3.62      | 26.21      | 0.827      | 0.219       | 2.88        | 28.53        | 0.857        | 0.196         | 17.81      | 12.60             |


**Notes**:
- Evaluation on an NVIDIA GeForce RTX 4090 GPU
- LPIPS uses VGG as the extractor.
- Mask-ShadowGAN and DC-ShadowNet are unsupervised methods.
- G2R-ShadowNet is a weakly-supervised method.
- The PyTorch implementation of [DSC](https://github.com/stevewongv/DSC-PyTorch/tree/master) for shadow removal based on [DSC (Caffe)](https://github.com/xw-hu/DSC).
- The PyTorch implementation of [ST-CGAN](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks?tab=readme-ov-file).

### Cross-dataset generalization evaluation. Trained on SRD and tested on DESOBA: [Results-Part1](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval_CrossVal-1.zip), [Results-Part2](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval_CrossVal-2.zip), [Results-Part3](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval_CrossVal-3.zip), [Results-Part4](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval_CrossVal-4.zip), [Results-Part5](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/ShadowRemoval_CrossVal-5.zip)

| Input Size | Metric | ST-CGAN (CVPR 2018) | SP+M-Net (ICCV 2019) | Mask-ShadowGAN (ICCV 2019) | DSC (TPAMI 2020) | Auto (CVPR 2021) | G2R-ShadowNet (CVPR 2021) | DC-ShadowNet (ICCV 2021) | BMNet (CVPR 2022) | SG-ShadowNet (ECCV 2022) | ShadowDiffusion (CVPR 2023) | ShadowFormer (AAAI 2023) | ShadowMaskFormer (arXiv 2024) | HomoFormer (CVPR 2024) |
|:----------:|:-------:|:-----------------:|:--------------------:|:-------------------------:|:---------------:|:---------------:|:-------------------------:|:-----------------------:|:----------------:|:-----------------------:|:-------------------------:|:----------------------:|:--------------------------:|:-----------------------:|
| 256x256    | RMSE (↓)   | 7.07              | 5.10                 | 6.94                      | 6.66            | 5.88            | 5.13                      | 6.88                    | 5.37             | 4.92                     | 5.59                      | 5.01                 | 5.82                       | 5.02                    |
| 256x256    | PSNR (↑)   | 20.23             | 23.35                | 20.47                     | 20.71           | 22.62           | 23.14                     | 20.58                   | 22.75            | 23.36                    | 22.08                     | 23.49                | 22.14                      | 23.41                   |
| 512x512    | RMSE (↓)   | 6.65              | 4.57                 | 6.74                      | 5.58            | 5.05            | 4.60                      | 6.62                    | 5.06             | 4.47                     | 5.50                      | 4.55                 | 5.51                       | 4.42                    |
| 512x512    | PSNR (↑)   | 20.98             | 24.80                | 20.96                     | 22.61           | 24.16           | 24.56                     | 21.25                   | 23.65            | 24.53                    | 22.34                     | 24.81                | 23.11                      | 24.89                   |


**Notes**:
- DESOBA only labels cast shadows and we set the self shadows on objects as "don’t care" in evaluation. 

### Datasets
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view?pli=1), [Test](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view), and [Masks](https://yuhaoliu7456.github.io/projects/RRL-Net/index.html)
- [ISTD+](https://drive.google.com/file/d/1rsCSWrotVnKFUqu9A_Nw9Uf-bJq_ryOv/view)
- [DESOBA](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA)

### Metrics
- MAE: see this GitHub repository.
- PSNE: see this GitHub repository.
- SSIM: see this GitHub repository.
- LPIPS: see this GitHub repository. 

## Document Shadow Removal

### Comparing document shadow removal methods on RDD: [Results](https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Results/DocShadowRemoval.zip) 

| Methods                                  | RMSE (↓)  | PSNR (↑)  | SSIM (↑)  | LPIPS (↓)  | Param.(M) | Infer. (images/s) |
|:----------------------------------------:|:---------:|:---------:|:---------:|:----------:|:---------:|:-----------------:|
| BEDSR-Net (CVPR 2020)                    | 3.13      | 28.480    | 0.912     | 0.171      | 32.21     | 10.41             |
| FSENet (ICCV 2023)                       | 2.46      | 31.251    | 0.948     | 0.161      | 29.40     | 19.37             |


**Notes**:
- Evaluation on an NVIDIA GeForce RTX 4090 GPU
- LPIPS uses VGG as the extractor.

### Dataset
- [RDD](https://github.com/hyyh1314/RDD)

## Bibtex
If you find our work, results, models, and unified evaluation code useful, please consider citing our paper as follows:
```
@article{hu2024unveiling,
  title={Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning},
  author={Hu, Xiaowei and Xing, Zhenghao and Wang, Tianyu and Fu, Chi-Wing and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2409.02108},
  year={2024}
}
```
