# DL-Project---Group-6

.
├── Models/                     # Saved models and weights
├── Notebooks/                 # Jupyter notebooks for experiments
│   ├── Baseline.ipynb         # TwoBranchInceptionV3 using RGB + Lab channels
│   ├── Improvement-1a.ipynb   # EfficientNet-B0 with 6-channel RGB-Lab input
│   ├── Improvement-1b.ipynb   # 1a + advanced image augmentations
│   └── improvement_2.ipynb    # 1b + CBAM attention modules and residual blocks
└── README.md



## Tomato Leaf Disease Classification using Deep Learning
This project addresses the critical need for accurate tomato leaf disease detection in real-world agricultural settings. Building upon recent deep learning advancements, we leverage the PlantDoc dataset—a more realistic alternative to the commonly used PlantVillage dataset—to train and evaluate robust disease classification models.

### Problem Overview
Tomato crops are economically important but vulnerable to a variety of diseases. Early and accurate disease identification can prevent large-scale losses. While prior works often rely on the idealized PlantVillage dataset, this project uses the PlantDoc dataset—featuring complex real-world backgrounds—to build models that generalize better to real farm conditions.

### Methodology
#### Baseline – TwoBranchInceptionV3
We use a dual-branch architecture that splits Lab color channels into luminance and chrominance branches. The output is fused and fed into a modified InceptionV3 for classification.

#### Improvement 1A – EfficientNet with RGB+Lab Input
We replace InceptionV3 with EfficientNet-B0, pretrained on ImageNet1K. The input is extended to 6 channels (RGB + Lab), and the first conv layer is adjusted accordingly.

#### Improvement 1B – Advanced Augmentation
To enhance generalization, we apply aggressive image augmentations (e.g., flips, jitter, perspective, blur) to the RGB channels while preserving Lab alignment.

#### Improvement 2 – CBAM Attention + Residual Learning
We add Convolutional Block Attention Modules (CBAM) and a residual block to EfficientNet to help the model focus on disease-relevant leaf regions and improve interpretability.

