# Unveiling-Deep-Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning

This repository contains the results and trained models for deep-learning methods used in shadow detection, removal, and generation, as presented in our paper "Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning." In this paper, we present a comprehensive survey of shadow detection, removal, and generation in both images and videos in the era of deep learning over the past decade, encompassing tasks, deep models, datasets, and evaluation metrics. Key contributions include a thorough investigation of overfitting challenges, meticulous scrutiny of dataset quality, in-depth exploration of relationships between model size, speed, and performance, and a comprehensive cross-dataset generalization study.

## Highlights
+ Comprehensive Survey of Shadow Analysis in the Deep Learning Era.
+ Fair Comparisons of the Existing Methods: unified platform + newly refined datasets with corrected noisy labels and ground-truth images.
+ Exploration of Model Size, Speed, and Performance Relationships: a more comprehensive comparison of different evaluation aspects.
+ Cross-Dataset Generalization Study: assess the generalization capability of deep models across diverse datasets.
+ Overview of Open Issues and Future Directions, Particularly with AIGC and Large Models.


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
|            | BDRAR (ECCV 2018)        | 5.44             | 8.42              | 42.46      | 31.34             |
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
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view?pli=1), [Testing](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view), and [Masks](https://yuhaoliu7456.github.io/projects/RRL-Net/index.html)

### Metrics
- BER: [Python]()

## Video Shadow Detection


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


**Notes**:
- Evaluation on NVIDIA GeForce RTX 4090 GPU

### Datasets
- [SOBA](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP)
- [SOBA-VID]()

### Metrics
- SOAP: [Python](https://github.com/stevewongv/SSIS)
- SOAP-VID: [Python]()

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
