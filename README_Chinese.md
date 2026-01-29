# GAN_MFS2

---

![License](https://img.shields.io/github/license/fl1pcoin/GAN_MFS2?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## 概述

GAN_MFS2 能够生成高质量的合成表格数据，这些数据与真实世界的数据集非常相似。它支持保护隐私的数据共享，通过可靠的数据增强来提升机器学习工作流，并为分析和建模任务提供可信的结果。

---

## 目录

- [核心功能](#core-features)
- [安装](#installation)
- [快速开始](#getting-started)
- [文档](#documentation)
- [贡献](#contributing)
- [许可证](#license)
- [引用](#citation)

---
## 核心功能

1. **WGAN-GP 实现**：实现带有梯度惩罚的Wasserstein GAN（WGAN-GP），通过使用Wasserstein距离并利用梯度惩罚强制满足Lipschitz连续性，确保训练稳定，从而生成更可靠且一致的合成数据。
2. **元特征统计（MFS）保留**：通过保留原始数据的元特征统计（MFS）来增强GAN，包括均值、方差、相关性、协方差、特征值和高阶矩，确保合成表格数据的高保真度。
3. **基于PyTorch的MFS计算**：提供元特征提取的完全可微、原生PyTorch实现（通过MFEToTorch），支持端到端的基于梯度的优化，并与GAN训练循环集成，实现实时MFS对齐。
4. **用于MFS对齐的Wasserstein距离**：使用最优传输（Wasserstein距离）来衡量并最小化真实数据与合成数据之间元特征分布的差异，提高生成样本的统计相似性和实用性。
5. **全面的评估指标**：包含一系列实用性和保真度指标，如R²、RMSE、MAPE、Jensen-Shannon散度、相关矩阵距离和拓扑数据分析，以严格评估合成数据的质量。
6. **残差网络生成器**：采用基于残差连接的生成器架构，在层间拼接输入，促进更深的网络结构并在训练期间改善梯度流动，从而更有效地生成合成数据。

---

## 安装

使用以下方法之一安装GAN_MFS2：

**从源码构建：**

1. 克隆GAN_MFS2仓库：
```sh
git clone https://github.com/fl1pcoin/GAN_MFS2
```

2. 进入项目目录：
```sh
cd GAN_MFS2
```
## 快速开始

要开始使用GAN_MFS2，请按照以下步骤在您的表格数据集上训练一个增强了元特征统计（MFS）的WGAN-GP模型。

### 1. 准备您的数据

确保您的数据为表格格式（例如CSV），并包含目标列。示例：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('your_data.csv')
y = data.pop('target').values
X = data.values

# Split and combine for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
training_data = np.hstack([X_train, y_train.reshape(-1, 1)])
```

### 2. 配置并训练模型

使用`TrainerModified`进行MFS增强训练：

```python
from wgan_gp.training import TrainerModified
from wgan_gp.models import Generator, Discriminator

# Define hyperparameters
learning_params = {
    'epochs': 1000,
    'learning_rate_D': 0.0001,
    'learning_rate_G': 0.0001,
    'batch_size': 64,
    'gp_weight': 10,
    'generator_dim': (64, 128, 64),
    'discriminator_dim': (64, 128, 64),
    'mfs_lambda': [0.1, 2.0],
    'subset_mfs': ['mean', 'var', 'eigenvalues'],
    'sample_number': 10,
    'sample_frac': 0.5,
}

# Initialize trainer and start training
trainer = TrainerModified(
    generator=Generator(**learning_params),
    discriminator=Discriminator(**learning_params),
    **learning_params
)
trainer.train(data_loader, epochs=1000, plot_freq=100)
```

### 3. 使用Aim监控训练

该项目支持通过Aim进行实验跟踪。启用方式如下：

```python
import aim
tracker = aim.Run(experiment="WGAN-GP")
tracker["hparams"] = learning_params
```

### 4. 评估合成数据

训练完成后，使用内置指标评估合成数据：

- **实用性**：R²、RMSE、MAPE
- **保真度**：相关矩阵距离、Jensen-Shannon散度

评估函数详见`utils.py`。

更多详细信息，请在本地构建文档：

```bash
mkdocs serve --config-file osa_mkdocs.yml
```

然后访问 `http://localhost:8000`。

---

## 文档

详细的GAN_MFS2说明请参见[此处](https://roman223.github.io/GAN_MFS/)。

---

## 贡献

- **[报告问题](https://github.com/fl1pcoin/GAN_MFS2/issues)**：提交发现的错误或为项目记录功能请求。

---

## 许可证

该项目受MIT许可证保护。更多详情，请参阅[LICENSE](https://github.com/fl1pcoin/GAN_MFS2/tree/main/LICENSE)文件。

---

## 引用

如果您使用此软件，请按以下方式引用。

### APA格式：

    fl1pcoin (2025). GAN_MFS2 repository [Computer software]. https://github.com/fl1pcoin/GAN_MFS2

### BibTeX格式：

    @misc{GAN_MFS2,

        author = {fl1pcoin},

        title = {GAN_MFS2 repository},

        year = {2025},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/fl1pcoin/GAN_MFS2.git}},

        url = {https://github.com/fl1pcoin/GAN_MFS2.git}

    }

---
