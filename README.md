# Unveiling-Deep-Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning

This repository contains the results and trained models for deep-learning methods used in shadow detection, removal, and generation, as presented in our paper "Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning." This paper presents a comprehensive survey of shadow detection, removal, and generation in images and videos within the deep learning landscape over the past decade, covering tasks, deep models, datasets, and evaluation metrics. Key contributions include a comprehensive survey of shadow analysis, standardization of experimental comparisons, exploration of the relationships among model size, speed, and performance, a cross-dataset generalization study, identification of open issues and future directions, and provision of publicly available resources to support further research.

## Highlights
+ A Comprehensive Survey of Shadow Analysis in the Deep Learning Era.
+ Fair Comparisons of the Existing Methods. Unified platform + newly refined datasets with corrected noisy labels and ground-truth images.
+ Exploration of Model Size, Speed, and Performance Relationships. A more comprehensive comparison of different evaluation aspects.
+ Cross-Dataset Generalization Study. Assess the generalization capability of deep models across diverse datasets.
+ Overview of Open Issues and Future Directions with AIGC and Large Models.
+ Publicly Available Results, Trained Models, and Evaluation Metrics.


## Image Shadow Detection

### Comparing image shadow detection methods on SBU-Refine and CUHK-Shadow: [Results]() \& [Models]()

| Input Size | Methods                                   | BER (SBU-Refine) | BER (CUHK-Shadow) | Params (M) | Infer. (images/s) |
|:----------:|:-----------------------------------------:|:----------------:|:-----------------:|:----------:|:-----------------:|
| 256x256    | DSC (CVPR 2018, TPAMI 2020) | 6.79             | 10.97             | 122.49     | 26.86             |
|            | BDRAR (ECCV 2018)         | 6.01             | 9.68              | 42.46      | 39.76             |
|            | DSDNet# (CVPR 2019)      | 5.33             | 8.23              | 58.16      | 37.53             |
|            | MTMT-Net$ (CVPR 2020)          | 6.30             | 8.64              | 44.13      | 34.04             |
|            | FDRNet (ICCV 2021)           | 5.64             | 14.39             | 10.77      | 41.39             |
|            | FSDNet* (TIP 2021)           | 7.16             | 9.93              | 4.39       | 150.99            |
|            | ECA (MM 2021)                | 7.08             | 8.58              | 157.76     | 27.55             |
|            | SDDNet (MM 2023)             | 5.39             | 8.66              | 15.02      | 36.73             |
| 512x512    | DSC (CVPR 2018, TPAMI 2020)   | 6.34             | 9.53              | 122.49     | 22.59             |
|            | BDRAR (ECCV 2018)        | 5.44             | 8.42              | 42.46      | 31.34               |
|            | DSDNet# (CVPR 2019)       | 4.98             | 7.58              | 58.16      | 32.69             |
|            | MTMT-Net$ (CVPR 2020)            | 5.77             | 8.03              | 44.13      | 28.75             |
|            | FDRNet (ICCV 2021)          | 5.39             | 6.58              | 10.77      | 35.00             |
|            | FSDNet* (TIP 2021)           | 6.80             | 8.84              | 4.39       | 134.47            |
|            | ECA (MM 2021)                 | 7.52             | 7.99              | 157.76     | 22.41             |
|            | SDDNet (MM 2023)             | 4.86             | 7.65              | 15.02      | 37.65             |

**Notes**:
- Evaluation on NVIDIA GeForce RTX 4090 GPU
- $: additional training data
- *: real-time shadow detector
- #: extra supervision from other methods

### Cross-dataset generalization evaluation. Trained on SBU-Refine and tested on SRD: [Results]()

| Input Size | Metric | DSC (CVPR 2018, TPAMI 2020) | BDRAR (ECCV 2018) | DSDNet# (CVPR 2019) | MTMT-Net$ (CVPR 2020) | FDRNet (ICCV 2021) | FSDNet* (TIP 2021) | ECA (MM 2021) | SDDNet (MM 2023) |
|:----------:|:-------:|:------------------------:|:---------------:|:-----------------:|:-------------------:|:----------------:|:----------------:|:-----------:|:--------------:|
| 256x256    | BER     | 11.10                    | 9.05            | 10.32             | 9.82                | 11.82            | 12.13            | 11.97        | 8.64          |
| 512x512    | BER     | 11.62                    | 8.37            | 8.88              | 9.08                | 8.81             | 11.94            | 12.71        | 7.65          |


### Datasets
- [SBU-Refine](https://github.com/hanyangclarence/SILT/releases/tag/refined_sbu)
- [CUHK-Shadow](https://github.com/xw-hu/CUHK-Shadow#cuhk-shadow-dateset)
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view?pli=1), [Test](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view), and [Masks](https://yuhaoliu7456.github.io/projects/RRL-Net/index.html)

### Metrics
- BER: see this GitHub repository.

## Video Shadow Detection
### Comparison of video shadow detection methods on ViSha: [Results]() \& [Models]()
| Methods                       | BER $\downarrow$ | IoU [%] $\uparrow$ | TS [%] $\uparrow$ | AVG $\uparrow$ | Param. (M) | Infer. (frames/s) |
|:-----------------------------:|:----------------:|:------------------:|:-----------------:|:--------------:|:----------:|:-----------------:|
| TVSD-Net (CVPR 2021) | 14.21            | 56.36              | 22.69             | 39.53          | 60.83      | 15.50             |
| STICT\$* (CVPR 2022)    | 13.05            | 43.75              | 39.10             | 41.43          | 26.17      | 91.34             |
| SC-Cor (ECCV 2022) | 12.80            | 55.56              | 23.68             | 39.62          | 58.16      | 27.91             |
| SCOTCH and SODA (CVPR 2023) | 10.36     | 61.24              | 25.76             | 43.50          | 53.11      | 16.16             |
| ShadowSAM (TCSVT 2023) | 13.38            | 61.72              | 23.77             | 42.75          | 93.74      | 15.53             |

**Notes**:
- Evaluation on NVIDIA GeForce RTX 4090 GPU
- $: additional training data
- *: real-time shadow detector

### Datasets
- [ViSha](https://erasernut.github.io/ViSha.html)

### Metrics
- TS: see this GitHub repository.
- IoU: see this GitHub repository.
- BER: see this GitHub repository.

  
## Instance Shadow Detection

### Comparing image instance shadow detection methods on the SOBA-testing set: [Results]() \& [Models]()

| Methods                        | $SOAP_{segm}$ | $SOAP_{bbox}$ | Asso. $AP_{segm}$ | Asso. $AP_{bbox}$ | Ins. $AP_{segm}$ | Ins. $AP_{bbox}$ | Param. (M) | Infer. (images/s) |
|:-------------------------------|:-------------:|:-------------:|:-----------------:|:-----------------:|:----------------:|:----------------:|:----------:|:-----------------:|
| LISA (CVPR 2020)       | 23.5          | 21.9          | 42.7              | 50.4              | 39.7             | 38.2             | 91.26      | 8.16              |
| SSIS (CVPR 2021)       | 29.9          | 26.8          | 52.3              | 59.2              | 43.5             | 41.5             | 57.87      | 5.83              |
| SSISv2 (TPAMI 2023)     | 35.3          | 29.0          | 59.2              | 63.0              | 50.2             | 44.4             | 76.77      | 5.17              |

### Comparing image instance shadow detection methods on the SOBA-challenge set: [Results]() 

| Methods                        | $SOAP_{segm}$ | $SOAP_{bbox}$ | Asso. $AP_{segm}$ | Asso. $AP_{bbox}$ | Ins. $AP_{segm}$ | Ins. $AP_{bbox}$ | Param. (M) | Infer. (images/s) |
|:-------------------------------|:-------------:|:-------------:|:-----------------:|:-----------------:|:----------------:|:----------------:|:----------:|:-----------------:|
| LISA (CVPR 2020)       | 10.2          | 9.8           | 21.6              | 26.0              | 23.9             | 24.7             | 91.26      | 4.52              |
| SSIS (CVPR 2021)       | 12.8          | 12.9          | 28.4              | 32.5              | 25.7             | 26.5             | 57.87      | 2.26              |
| SSISv2 (TPAMI 2023)     | 17.7          | 15.0          | 34.5              | 37.2              | 31.0             | 28.4             | 76.77      | 1.91              |


### Cross-dataset generalization evaluation. Trained on SOBA and tested on SOBA-VID: [Results]()

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

### Comparing image shadow removal methods on SRD and ISTD+: [Results]() \& [Models]()

| Input Size | Methods                                | MAE (SRD) | PSNR (SRD) | SSIM (SRD) | LPIPS (SRD) | MAE (ISTD+) | PSNR (ISTD+) | SSIM (ISTD+) | LPIPS (ISTD+) | Params (M) | Infer. (images/s) |
|:----------:|:--------------------------------------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:------------:|:------------:|:-------------:|:----------:|:-----------------:|
| -          | input                                  | 9.47      | 17.91      | 0.801      | 0.194       | 7.16        | 20.20        | 0.885        | 0.107         | -          | -                 |
| 256x256    | ST-CGAN (CVPR 2018)                    | 8.91      | 18.02      | 0.389      | 0.567       | 7.43        | 20.04        | 0.387        | 0.497         | 58.49      | 71.69             |
|            | SP+M-Net (ICCV 2019)                   | 6.46      | 20.71      | 0.641      | 0.441       | 4.64        | 23.36        | 0.714        | 0.376         | 54.42      | 33.88             |
|            | Mask-ShadowGAN (ICCV 2019)             | 4.32      | 24.67      | 0.662      | 0.427       | 3.70        | 25.50        | 0.720        | 0.377         | 22.76      | 64.77             |
|            | DSC (TPAMI 2020)                       | 5.36      | 22.59      | 0.650      | 0.422       | 5.59        | 23.07        | 0.679        | 0.395         | 122.49     | 46.75             |
|            | Auto (CVPR 2021)                       | 5.37      | 23.20      | 0.694      | 0.370       | 3.53        | 26.10        | 0.718        | 0.365         | 196.76     | 33.23             |
|            | G2R-ShadowNet (CVPR 2021)              | 6.08      | 21.72      | 0.619      | 0.460       | 4.37        | 24.23        | 0.696        | 0.396         | 22.76      | 3.62              |
|            | DC-ShadowNet (ICCV 2021)               | 4.27      | 24.72      | 0.670      | 0.383       | 3.89        | 25.18        | 0.693        | 0.406         | 10.59      | 40.51             |
|            | BMNet (CVPR 2022)                      | 4.39      | 24.24      | 0.721      | 0.327       | 3.34        | 26.62        | 0.731        | 0.354         | 0.58       | 17.42             |
|            | SG-ShadowNet (ECCV 2022)               | 4.60      | 24.10      | 0.636      | 0.443       | 3.32        | 26.80        | 0.717        | 0.369         | 6.17       | 16.51             |
|            | ShadowDiffusion (CVPR 2023)            | 4.84      | 23.26      | 0.684      | 0.363       | 3.44        | 26.51        | 0.688        | 0.404         | 55.52      | 9.73              |
|            | ShadowFormer (AAAI 2023)               | 4.44      | 24.28      | 0.715      | 0.348       | 3.45        | 26.55        | 0.728        | 0.350         | 11.37      | 32.57             |
|            | ShadowMaskFormer (arXiv 2024)          | 4.69      | 23.85      | 0.671      | 0.386       | 3.39        | 26.57        | 0.698        | 0.395         | 2.28       | 17.63             |
|            | HomoFormer (CVPR 2024)                 | 4.17      | 24.64      | 0.723      | 0.325       | 3.37        | 26.72        | 0.732        | 0.348         | 17.81      | 16.14             |
| 512x512    | ST-CGAN (CVPR 2018)                    | 9.26      | 17.63      | 0.363      | 0.504       | 7.25        | 20.40        | 0.417        | 0.567         | 58.49      | 42.89             |
|            | SP+M-Net (ICCV 2019)                   | 5.95      | 21.56      | 0.779      | 0.280       | 4.29        | 24.29        | 0.860        | 0.190         | 54.42      | 24.48             |
|            | Mask-ShadowGAN (ICCV 2019)             | 3.83      | 25.98      | 0.803      | 0.270       | 3.42        | 26.51        | 0.865        | 0.196         | 22.76      | 52.70             |
|            | DSC (TPAMI 2020)                       | 5.01      | 23.88      | 0.752      | 0.311       | 5.37        | 24.15        | 0.801        | 0.260         | 122.49     | 37.80             |
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
- Evaluation on NVIDIA GeForce RTX 4090 GPU
- LPIPS uses VGG as the extractor.
- Mask-ShadowGAN and DC-ShadowNet are unsupervised methods.
- G2R-ShadowNet is a weakly-supervised method.

### Cross-dataset generalization evaluation. Trained on SRD and tested on DESOBA: [Results]()

| Input Size | Metric | ST-CGAN (CVPR 2018) | SP+M-Net (ICCV 2019) | Mask-ShadowGAN (ICCV 2019) | DSC (TPAMI 2020) | Auto (CVPR 2021) | G2R-ShadowNet (CVPR 2021) | DC-ShadowNet (ICCV 2021) | BMNet (CVPR 2022) | SG-ShadowNet (ECCV 2022) | ShadowDiffusion (CVPR 2023) | ShadowFormer (AAAI 2023) | ShadowMaskFormer (arXiv 2024) | HomoFormer (CVPR 2024) |
|:----------:|:-------:|:-----------------:|:--------------------:|:-------------------------:|:---------------:|:---------------:|:-------------------------:|:-----------------------:|:----------------:|:-----------------------:|:-------------------------:|:----------------------:|:--------------------------:|:-----------------------:|
| 256x256    | MAE     | 12.28             | 5.51                 | 6.94                      | 9.90            | 5.88            | 5.13                      | 6.88                    | 5.37             | 4.92                     | 5.59                      | 5.01                 | 5.82                       | 5.02                    |
| 256x256    | PSNR    | 15.38             | 22.65                | 20.47                     | 17.90           | 22.62           | 23.14                     | 20.58                   | 22.75            | 23.36                    | 22.08                     | 23.49                | 22.14                      | 23.41                   |
| 512x512    | MAE     | 12.51             | 5.06                 | 6.74                      | 8.82            | 5.05            | 4.60                      | 6.62                    | 5.06             | 4.47                     | 5.50                      | 4.55                 | 5.51                       | 4.42                    |
| 512x512    | PSNR    | 15.18             | 23.87                | 20.96                     | 19.08           | 24.16           | 24.56                     | 21.25                   | 23.65            | 24.53                    | 22.34                     | 24.81                | 23.11                      | 24.89                   |


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
  

## Bibtex
If you find our work, models or results useful, please consider citing our paper as follows:
```
@article{hu2024unveiling,
  title={Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning},
  author={Hu, Xiaowei and Xing, Zhenghao and Wang, Tianyu and Fu, Chi-Wing and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:24},
  year={2024}
}
```
