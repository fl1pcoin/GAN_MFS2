# GAN_MFS2
---

![License](https://img.shields.io/github/license/fl1pcoin/GAN_MFS2?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black)

---

## Overview

GAN_MFS2 generates realistic synthetic tabular data that preserves key statistical traits, enabling privacy‑safe sharing and improved machine learning models. It delivers high‑fidelity, useful synthetic datasets for analytics and modeling while ensuring trustworthy results.

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

1. **Wasserstein GAN with Gradient Penalty (WGAN‑GP)**: Stable adversarial training using the Wasserstein distance and a gradient penalty to enforce Lipschitz continuity, resulting in reliable synthetic tabular data generation.
2. **Meta‑Feature Statistics (MFS) Preservation**: Maintains key statistical properties (mean, variance, correlation, covariance, eigenvalues, higher‑order moments) of the original dataset during training, ensuring high fidelity of synthetic samples.
3. **PyTorch‑native MFS Computation**: Differentiable extraction of meta‑features via MFEToTorch, allowing end‑to‑end gradient‑based optimization and real‑time alignment of MFS within the GAN loop.
4. **Wasserstein Distance for MFS Alignment**: Uses optimal transport to quantify and minimize the discrepancy between real and synthetic meta‑feature distributions, improving statistical similarity and downstream utility.
5. **Residual Connection Generator**: A generator architecture that concatenates intermediate representations, enabling deeper networks and better gradient flow for complex tabular data synthesis.
6. **Comprehensive Evaluation Suite**: Includes utility metrics (R², RMSE, MAPE), fidelity metrics (Jensen‑Shannon divergence, correlation matrix distance, topological distance), and cosine distance of correlation matrices to rigorously assess synthetic data quality.

---

## Installation

Install GAN_MFS2 using one of the following methods:

**Build from source**:

1. Clone the GAN_MFS2 repository:
   ```sh
   git clone https://github.com/fl1pcoin/GAN_MFS2
   ```
2. Navigate to the project directory:
   ```sh
   cd GAN_MFS2
   ```
3. Install the project dependencies:
   ```sh
   pip install -r requirements.txt
   ```

---

## Getting Started

### 1. Prepare Your Data
Ensure your dataset is a CSV with a target column. Example:

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
Use `TrainerModified` for MFS‑enhanced training:

```python
from wgan_gp.training import TrainerModified
from wgan_gp.models import Generator, Discriminator

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

trainer = TrainerModified(
    generator=Generator(**learning_params),
    discriminator=Discriminator(**learning_params),
    **learning_params
)
trainer.train(data_loader, epochs=1000, plot_freq=100)
```

### 3. Monitor Training with Aim

```python
import aim
tracker = aim.Run(experiment="WGAN-GP")
tracker["hparams"] = learning_params
```

### 4. Evaluate Synthetic Data
After training, use the built‑in metrics (R², RMSE, MAPE, correlation matrix distance, Jensen‑Shannon divergence) found in `utils.py`.

For a full local documentation view:

```bash
mkdocs serve --config-file osa_mkdocs.yml
```

Visit `http://localhost:8000` to explore the docs.

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

fl1pcoin (2025). GAN_MFS2 repository [Computer software]. https://github.com/fl1pcoin/GAN_MFS2

@misc{GAN_MFS2,
    author = {fl1pcoin},
    title = {GAN_MFS2 repository},
    year = {2025},
    publisher = {github.com},
    journal = {github.com repository},
    howpublished = {\url{https://github.com/fl1pcoin/GAN_MFS2.git}},
    url = {https://github.com/fl1pcoin/GAN_MFS2.git}
}
