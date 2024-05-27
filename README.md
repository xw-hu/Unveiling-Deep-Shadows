# Unveiling-Deep-Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning

This repository contains the codebase for deep models used in shadow detection, removal, and generation, as presented in our paper "Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning." In this paper, we present a comprehensive survey of shadow detection, removal, and generation in both images and videos in the era of deep learning over the past decade, encompassing tasks, deep models, datasets, and evaluation metrics. Key contributions include a thorough investigation of overfitting challenges, meticulous scrutiny of dataset quality, in-depth exploration of relationships between model size, speed, and performance, and a comprehensive cross-dataset generalization study.

## Highlights
+ Comprehensive Survey of Shadow Analysis in the Deep Learning Era.
+ Fair Comparisons of the Existing Methods: unified platform + newly refined datasets with corrected noisy labels and ground-truth images.
+ Exploration of Model Size, Speed, and Performance Relationships: a more comprehensive comparison of different evaluation aspects.
+ Cross-Dataset Generalization Study: assess the generalization capability of deep models across diverse datasets.
+ Overview of Open Issues and Future Directions, Particularly with AIGC and Large Models.


## Image Shadow Detection

### Shadow Detection Evaluation on SBU-Refine and CUHKShadow

| Input Size | Methods                                   | BER (SBU-Refine) | BER (CUHK-Shadow) | Params (M) | Infer. (images/s) |
|:----------:|:-----------------------------------------:|:----------------:|:-----------------:|:----------:|:-----------------:|
| 256x256    | DSC (CVPR'18, TPAMI'20) | 6.79             | 10.97             | 122.49     | 26.86             |
|            | BDRAR (ECCV'18)         | 6.01             | 9.68              | 42.46      | 39.76             |
|            | DSDNet# (CVPR'19)      | 5.33             | 8.23              | 58.16      | 37.53             |
|            | MTMT-Net$ (CVPR'20)          | 6.30             | 8.64              | 44.13      | 34.04             |
|            | FDRNet (ICCV'21)           | 5.64             | 14.39             | 10.77      | 41.39             |
|            | FSDNet* (TIP'21)           | 7.16             | 9.93              | 4.39       | 150.99            |
|            | ECA (MM'21)                | 7.08             | 8.58              | 157.76     | 27.55             |
|            | SDDNet (MM'23)             | 5.39             | 8.66              | 15.02      | 36.73             |
| 512x512    | DSC (CVPR'18, TPAMI'20)   | 6.34             | 9.53              | 122.49     | 22.59             |
|            | BDRAR (ECCV'18)        | 5.44             | 8.42              | 42.46      | 31.34             |
|            | DSDNet# (CVPR'19)       | 4.98             | 7.58              | 58.16      | 32.69             |
|            | MTMT-Net$ (CVPR'20)            | 5.77             | 8.03              | 44.13      | 28.75             |
|            | FDRNet (ICCV'21)          | 5.39             | 6.58              | 10.77      | 35.00             |
|            | FSDNet* (TIP'21)           | 6.80             | 8.84              | 4.39       | 134.47            |
|            | ECA (MM'21)                 | 7.52             | 7.99              | 157.76     | 22.41             |
|            | SDDNet (MM'23)             | 4.86             | 7.65              | 15.02      | 37.65             |

**Notes**:
- Evaluation on NVIDIA GeForce RTX 4090 GPU
- $: additional training data
- *: real-time shadow detector
- #: extra supervision from other methods


## Video Shadow Detection


## Instance Shadow Detection


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
