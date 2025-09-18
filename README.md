![raw](https://github.com/user-attachments/assets/924844ef-2536-4bde-a330-5e30f6b0762c)

# WarpGBM

WarpGBM is a high-performance, GPU-accelerated Gradient Boosted Decision Tree (GBDT) library built with PyTorch and CUDA. It offers blazing-fast histogram-based training and efficient prediction, with compatibility for research and production workflows.

**New in v1.0.0:** WarpGBM introduces *Invariant Gradient Boosting* — a powerful approach to learning signals that remain stable across shifting environments (e.g., time, regimes, or datasets). Powered by a novel algorithm called **[Directional Era-Splitting (DES)](https://arxiv.org/abs/2309.14496)**, WarpGBM doesn't just train faster than other leading GBDT libraries — it trains smarter.

If your data evolves over time, WarpGBM is the only GBDT library designed to *adapt and generalize*.
---

## Contents

- [Features](#features)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Learning Invariant Signals Across Environments](#learning-invariant-signals-across-environments)
  - [Why This Matters](#why-this-matters)
  - [Visual Intuition](#visual-intuition)
  - [Key References](#key-references)
- [Examples](#examples)
  - [Quick Comparison with LightGBM CPU version](#quick-comparison-with-lightgbm-cpu-version)
  - [Pre-binned Data Example (Numerai)](#pre-binned-data-example-numerai)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [Version Notes](#version-notes)


## Features

- **Blazing-fast GPU training** with custom CUDA kernels for binning, histogram building, split finding, and prediction
- **Invariant signal learning** via [Directional Era-Splitting (DES)](https://arxiv.org/abs/2309.14496) — designed for datasets with shifting environments (e.g., time, regimes, experimental settings)
- Drop-in **scikit-learn style interface** for easy adoption
- Supports **pre-binned data** or **automatic quantile binning**
- Works with `float32` or `int8` inputs
- Built-in **validation and early stopping** support with MSE, RMSLE, or correlation metrics
- Supports continous learning
- Simple install with `pip`, no custom drivers required

> 💡 **Note:** WarpGBM v1.0.0 is a *generalization* of the traditional GBDT algorithm.
> To run standard GBM training at maximum speed, simply omit the `era_id` argument — WarpGBM will behave like a traditional booster but with industry-leading performance.

---

## Benchmarks

### Scikit-Learn Synthetic Data: 1 Million Rows and 1,000 Features

In this benchmark we compare the speed and in-sample correlation of **WarpGBM v1.0.0** against LightGBM, XGBoost and CatBoost, all with their GPU-enabled versions. This benchmark runs on Google Colab with the L4 GPU environment.

```
   WarpGBM:   corr = 0.8882, train = 17.4s, infer = 3.2s
   XGBoost:   corr = 0.8877, train = 33.2s, infer = 8.0s
  LightGBM:   corr = 0.8604, train = 29.8s, infer = 1.6s
  CatBoost:   corr = 0.8935, train = 392.1s, infer = 379.2s
```

Colab Notebook: https://colab.research.google.com/drive/16U1kbYlD5HibGbnF5NGsjChZ1p1IA2pK?usp=sharing

---

## Installation

### Recommended (GitHub, always latest):

```bash
pip install git+https://github.com/jefferythewind/warpgbm.git
```

This installs the latest version directly from GitHub and compiles CUDA extensions on your machine using your **local PyTorch and CUDA setup**. It's the most reliable method for ensuring compatibility and staying up to date with the latest features.

### Alternatively (PyPI, stable releases):

```bash
pip install warpgbm
```

This installs from PyPI and also compiles CUDA code locally during installation. This method works well **if your environment already has PyTorch with GPU support** installed and configured.

> **Tip:**\
> If you encounter an error related to mismatched or missing CUDA versions, try installing with the following flag. This is currently required in the Colab environments.
>
> ```bash
> pip install warpgbm --no-build-isolation
> ```

### Windows

Thank you, ShatteredX, for providing working instructions for a Windows installation.

```
git clone https://github.com/jefferythewind/warpgbm.git
cd warpgbm
python setup.py bdist_wheel
pip install .\dist\warpgbm-0.1.15-cp310-cp310-win_amd64.whl
```

Before either method, make sure you’ve installed PyTorch with GPU support:\
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## Learning Invariant Signals Across Environments

Most supervised learning models rely on an assumption known as the **Empirical Risk Minimization (ERM)** principle. Under ERM, the data distribution connecting inputs \( X \) and targets \( Y \) is assumed to be **fixed** and **stationary** across training, validation, and test splits. That is:

> The patterns you learn from the training set are expected to generalize out-of-sample — *as long as the test data follows the same distribution as the training data.*

However, this assumption is often violated in real-world settings. Data frequently shifts across time, geography, experimental conditions, or other hidden factors. This phenomenon is known as **distribution shift**, and it leads to models that perform well in-sample but fail catastrophically out-of-sample.

This challenge motivates the field of **Out-of-Distribution (OOD) Generalization**, which assumes your data is drawn from **distinct environments or eras** — e.g., time periods, customer segments, experimental trials. Some signals may appear predictive within specific environments but vanish or reverse in others. These are called **spurious signals**. On the other hand, signals that remain consistently predictive across all environments are called **invariant signals**.

WarpGBM v1.0.0 introduces **Directional Era-Splitting (DES)**, a new algorithm designed to identify and learn from invariant signals — ignoring signals that fail to generalize across environments.

---

### Why This Matters

- Standard models trained via ERM can learn to exploit **spurious correlations** that only hold in some parts of the data.
- DES explicitly tests whether a feature's split is **directionally consistent** across all eras — only such *invariant splits* are kept.
- This approach has been shown to reduce overfitting and improve out-of-sample generalization, particularly in financial and scientific datasets.

---

### Visual Intuition

We contrast two views of the data:

- **ERM Setting**: All data is assumed to come from the same source (single distribution).\
  No awareness of environments — spurious signals can dominate.

- **OOD Setting (Era-Splitting)**: Data is explicitly grouped by environment (era).\
  The model checks whether a signal holds across all groups — enforcing **robustness**.

<img src="https://github.com/user-attachments/assets/2be11ef3-6f2e-4636-ab91-307a73add247" alt="ChatGPT Image May 28, 2025, 05_05_09 PM" width="320"/>




---

### Key References

- **Invariant Risk Minimization (IRM)**: [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893)
- **Learning Explanations That Are Hard to Vary**: [Parascandolo et al., 2020](https://arxiv.org/abs/2009.00329)
- **Era Splitting: Invariant Learning for Decision Trees**: [DeLise, 2023](https://arxiv.org/abs/2309.14496)

---

WarpGBM is the **first open-source GBDT framework to integrate this OOD-aware approach natively**, using efficient CUDA kernels to evaluate per-era consistency during tree growth. It’s not just faster — it’s smarter.

---

## Examples

WarpGBM is easy to drop into any supervised learning workflow and comes with curated examples in the `examples/` folder.

 - `Spiral Data.ipynb`: synthetic OOD benchmark from Learning Explanations That Are Hard to Vary

### Quick Comparison with LightGBM CPU version

```python
import numpy as np
from sklearn.datasets import make_regression
from time import time
import lightgbm as lgb
from warpgbm import WarpGBM

# Create synthetic regression dataset
X, y = make_regression(n_samples=100_000, n_features=500, noise=0.1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)

# Train LightGBM
start = time()
lgb_model = lgb.LGBMRegressor(max_depth=5, n_estimators=100, learning_rate=0.01, max_bin=7)
lgb_model.fit(X, y)
lgb_time = time() - start
lgb_preds = lgb_model.predict(X)

# Train WarpGBM
start = time()
wgbm_model = WarpGBM(max_depth=5, n_estimators=100, learning_rate=0.01, num_bins=7)
wgbm_model.fit(X, y)
wgbm_time = time() - start
wgbm_preds = wgbm_model.predict(X)

# Results
print(f"LightGBM:   corr = {np.corrcoef(lgb_preds, y)[0,1]:.4f}, time = {lgb_time:.2f}s")
print(f"WarpGBM:     corr = {np.corrcoef(wgbm_preds, y)[0,1]:.4f}, time = {wgbm_time:.2f}s")
```

**Results (Ryzen 9 CPU, NVIDIA 3090 GPU):**

```
LightGBM:   corr = 0.8742, time = 37.33s
WarpGBM:     corr = 0.8621, time = 5.40s
```

---

### Continous learning

```python
import numpy as np
from sklearn.datasets import make_regression
from warpgbm import WarpGBM

# Create synthetic regression dataset
X, y = make_regression(n_samples=100_000, n_features=500, noise=0.1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)

# Train WarpGBM inital model
wgbm_model = WarpGBM(max_depth=5, n_estimators=100, learning_rate=0.01, num_bins=7)
wgbm_model.fit(X, y)
wgbm_model.save_model("ckpt_100.pkl")

# Continue training of WarpGBM inital model
wgbm_model = WarpGBM(n_estimators=100)
wgbm_model.fit(X, y, init_model="ckpt_100.pkl")

# Results
print('WarpGBN number of trees: ', wgbm_model.n_estimators)

---

### Pre-binned Data Example (Numerai)

WarpGBM can save additional training time if your dataset is already pre-binned. The Numerai tournament data is a great example:

```python
import pandas as pd
from numerapi import NumerAPI
from time import time
import lightgbm as lgb
from warpgbm import WarpGBM
import numpy as np

napi = NumerAPI()
napi.download_dataset('v5.0/train.parquet', 'train.parquet')
train = pd.read_parquet('train.parquet')

feature_set = [f for f in train.columns if 'feature' in f]
target = 'target_cyrus'

X_np = train[feature_set].astype('int8').values
Y_np = train[target].values

# LightGBM
start = time()
lgb_model = lgb.LGBMRegressor(max_depth=5, n_estimators=100, learning_rate=0.01, max_bin=7)
lgb_model.fit(X_np, Y_np)
lgb_time = time() - start
lgb_preds = lgb_model.predict(X_np)

# WarpGBM
start = time()
wgbm_model = WarpGBM(max_depth=5, n_estimators=100, learning_rate=0.01, num_bins=7)
wgbm_model.fit(X_np, Y_np)
wgbm_time = time() - start
wgbm_preds = wgbm_model.predict(X_np)

# Results
print(f"LightGBM:   corr = {np.corrcoef(lgb_preds, Y_np)[0,1]:.4f}, time = {lgb_time:.2f}s")
print(f"WarpGBM:     corr = {np.corrcoef(wgbm_preds, Y_np)[0,1]:.4f}, time = {wgbm_time:.2f}s")
```

**Results (Google Colab Pro, A100 GPU):**

```
LightGBM:   corr = 0.0703, time = 643.88s
WarpGBM:     corr = 0.0660, time = 49.16s
```

---

## Documentation

### `WarpGBM` Parameters:
- `num_bins`: Number of histogram bins to use (default: 10)
- `max_depth`: Maximum depth of trees (default: 3)
- `learning_rate`: Shrinkage rate applied to leaf outputs (default: 0.1)
- `n_estimators`: Number of boosting iterations (default: 100)
- `min_child_weight`: Minimum sum of instance weight needed in a child (default: 20)
- `min_split_gain`: Minimum loss reduction required to make a further partition (default: 0.0)
- `histogram_computer`: Choice of histogram kernel (`'hist1'`, `'hist2'`, `'hist3'`) (default: `'hist3'`)
- `threads_per_block`: CUDA threads per block (default: 32)
- `rows_per_thread`: Number of training rows processed per thread (default: 4)
- `L2_reg`: L2 regularizer (default: 1e-6)
- `colsample_bytree`: Proportion of features to subsample to grow each tree (default: 1)

### Methods:
```
.fit(
   X,                             # numpy array (float or int) 2 dimensions (num_samples, num_features)
   y,                             # numpy array (float or int) 1 dimension (num_samples)
   era_id=None,                   # numpy array (int) 1 dimension (num_samples)
   X_eval=None,                   # numpy array (float or int) 2 dimensions (eval_num_samples, num_features) 
   y_eval=None,                   # numpy array (float or int) 1 dimension (eval_num_samples)
   eval_every_n_trees=None,       # const (int) >= 1 
   early_stopping_rounds=None,    # const (int) >= 1
   eval_metric='mse'              # string, one of 'mse', 'rmsle' or 'corr'. For corr, loss is 1 - correlation(y_true, preds)
)
```
Train with optional validation set and early stopping.


```
.predict(
   X                              # numpy array (float or int) 2 dimensions (predict_num_samples, num_features)
)
```
Predict on new data, using parallelized CUDA kernel.

---

## Acknowledgements

WarpGBM builds on the shoulders of PyTorch, scikit-learn, LightGBM, and the CUDA ecosystem. Thanks to all contributors in the GBDT research and engineering space.

---

## Version Notes

### v0.1.21

- Vectorized predict function replaced with CUDA kernel (`warpgbm/cuda/predict.cu`), parallelizing per sample, per tree.

### v0.1.23

- Adjust gain in split kernel and added support for an eval set with early stopping based on MSE.

### v0.1.25

- Added `colsample_bytree` parameter and new test using Numerai data.

### v0.1.26

- Fix Memory bugs in prediction and colsample bytree logic. Added "corr" eval metric. 

### v1.0.0

- Introduce invariant learning via directional era splitting (DES). Also streamline VRAM improvements over previous sub versions.
