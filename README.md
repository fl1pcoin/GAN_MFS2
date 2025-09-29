# GAN_MFS2

---

![License](https://img.shields.io/github/license/fl1pcoin/GAN_MFS2?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

GAN_MFS2 enables the creation of high-quality synthetic tabular data that closely resembles real-world datasets. It supports privacy-preserving data sharing, enhances machine learning workflows through reliable data augmentation, and ensures trustworthy results for analytics and modeling tasks.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---
## Core features

1. **WGAN-GP Implementation**: Implements Wasserstein GAN with Gradient Penalty (WGAN-GP) to ensure stable training by using the Wasserstein distance and enforcing Lipschitz continuity via gradient penalty, leading to more reliable and consistent synthetic data generation.
2. **Meta-Feature Statistics (MFS) Preservation**: Enhances the GAN by preserving statistical properties of the original data through Meta-Feature Statistics (MFS), including mean, variance, correlation, covariance, eigenvalues, and higher-order moments, ensuring high fidelity in synthetic tabular data.
3. **PyTorch-Based MFS Computation**: Provides a fully differentiable, PyTorch-native implementation of meta-feature extraction (via MFEToTorch), enabling end-to-end gradient-based optimization and integration with the GAN training loop for real-time MFS alignment.
4. **Wasserstein Distance for MFS Alignment**: Uses optimal transport (Wasserstein distance) to measure and minimize the discrepancy between real and synthetic data's meta-feature distributions, improving the statistical similarity and utility of generated samples.
5. **Comprehensive Evaluation Metrics**: Includes a suite of utility and fidelity metrics such as R², RMSE, MAPE, Jensen-Shannon divergence, correlation matrix distance, and topological data analysis to rigorously assess synthetic data quality.
6. **Residual Network Generator**: Employs a residual connection-based generator architecture that concatenates inputs across layers, facilitating deeper networks and improved gradient flow during training for more effective synthetic data generation.

---

## Installation

Install GAN_MFS2 using one of the following methods:

**Build from source:**

1. Clone the GAN_MFS2 repository:
```sh
git clone https://github.com/fl1pcoin/GAN_MFS2
```

2. Navigate to the project directory:
```sh
cd GAN_MFS2
```
## Getting Started

To get started with GAN_MFS2, follow these steps to train a WGAN-GP model enhanced with Meta-Feature Statistics (MFS) on your tabular dataset.

### 1. Prepare Your Data

Ensure your data is in a tabular format (e.g., CSV) with a target column. Example:

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

### 2. Configure and Train the Model

Use `TrainerModified` for MFS-enhanced training:

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

### 3. Monitor Training with Aim

The project supports experiment tracking via Aim. To enable:

```python
import aim
tracker = aim.Run(experiment="WGAN-GP")
tracker["hparams"] = learning_params
```

### 4. Evaluate Synthetic Data

After training, evaluate the synthetic data using built-in metrics:

- **Utility**: R², RMSE, MAPE
- **Fidelity**: Correlation matrix distance, Jensen-Shannon divergence

Refer to `utils.py` for evaluation functions.

For more details, build the documentation locally:

```bash
mkdocs serve --config-file osa_mkdocs.yml
```

Then visit `http://localhost:8000`.

---

## Documentation

A detailed GAN_MFS2 description is available [here](https://roman223.github.io/GAN_MFS/).

---

## Contributing

- **[Report Issues](https://github.com/fl1pcoin/GAN_MFS2/issues)**: Submit bugs found or log feature requests for the project.

---

## License

This project is protected under the MIT License. For more details, refer to the [LICENSE](https://github.com/fl1pcoin/GAN_MFS2/tree/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    fl1pcoin (2025). GAN_MFS2 repository [Computer software]. https://github.com/fl1pcoin/GAN_MFS2

### BibTeX format:

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
