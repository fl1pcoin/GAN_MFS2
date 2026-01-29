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

GAN_MFS2 генерирует реалистичные синтетические табличные данные, сохраняющие ключевые статистические свойства, что позволяет безопасно делиться данными и улучшать модели машинного обучения. Он обеспечивает высококачественные, полезные синтетические наборы данных для аналитики и моделирования, при этом гарантируя надёжные результаты.

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

1. **Wasserstein GAN with Gradient Penalty (WGAN‑GP)**: стабильная состязательная тренировка с использованием расстояния Вассерштейна и градиентного штрафа для обеспечения липшицевости, что приводит к надёжной генерации синтетических табличных данных.
2. **Meta‑Feature Statistics (MFS) Preservation**: сохраняет ключевые статистические свойства (среднее, дисперсию, корреляцию, ковариацию, собственные значения, моменты более высокого порядка) исходного набора данных во время обучения, обеспечивая высокую достоверность синтетических образцов.
3. **PyTorch‑native MFS Computation**: дифференцируемый извлечение мета‑свойств через MFEToTorch, позволяющий оптимизацию от градиента от начала до конца и синхронизацию MFS в реальном времени внутри цикла GAN.
4. **Wasserstein Distance for MFS Alignment**: использует оптимальный транспорт для количественной оценки и минимизации разницы между реальными и синтетическими распределениями мета‑свойств, улучшая статистическую схожесть и последующую полезность.
5. **Residual Connection Generator**: архитектура генератора, которая объединяет промежуточные представления, позволяя строить более глубокие сети и улучшать поток градиентов при синтезе сложных табличных данных.
6. **Comprehensive Evaluation Suite**: включает метрики полезности (R², RMSE, MAPE), метрики достоверности (дисперсия Дженсена‑Шеннона, расстояние матрицы корреляций, топологическое расстояние) и косинусное расстояние матриц корреляций для строгой оценки качества синтетических данных.

---

## Installation

Установите GAN_MFS2, используя один из следующих методов:

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
Убедитесь, что ваш набор данных в формате CSV с целевой колонкой. Пример:

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
Используйте `TrainerModified` для обучения с улучшением MFS:

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
После обучения используйте встроенные метрики (R², RMSE, MAPE, расстояние матрицы корреляций, дивергенция Дженсена‑Шеннона), которые находятся в `utils.py`.

Для полного локального просмотра документации:

```bash
mkdocs serve --config-file osa_mkdocs.yml
```

Посетите `http://localhost:8000`, чтобы изучить документацию.

---

## Documentation

Подробное описание GAN_MFS2 доступно [здесь](https://roman223.github.io/GAN_MFS/).

---

## Contributing

- **[Сообщить об ошибках](https://github.com/fl1pcoin/GAN_MFS2/issues)**: Отправляйте найденные ошибки или запросы на новые функции для проекта.

---

## License

Этот проект защищён лицензией MIT. Для более подробной информации смотрите файл [LICENSE](https://github.com/fl1pcoin/GAN_MFS2/tree/main/LICENSE).

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
