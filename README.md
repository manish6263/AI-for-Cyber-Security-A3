# Assignment 3: Defenses for Robust CNNs on CIFAR-10

**Course:** COL865 — Artificial Intelligence for Cybersecurity  
**Student:** Manish Patel
**Platform Used:** Kaggle (GPU T4 / A100)
**Framework:** PyTorch (torchvision + lpips + sklearn)

---

## 🧠 Overview

This project implements **four defense strategies** against common attacks on deep learning models trained with CIFAR-10:

| Threat Type | Defense Implemented | Output CSV/Image |
|--------------|---------------------|------------------|
| Adversarial Attack (FGSM) | Adversarial Training | `defense_adversarial.csv` |
| Training Set Poisoning | Spectral Signature Filtering | `defense_poison.csv` |
| Membership Inference | Label Smoothing + Temperature Scaling | `defense_mi.csv` |
| Model Inversion | Gradient Regularization | `defense_inversion.csv`, `inversion_defended.png` |

Each script is modular, self-contained, and can be executed independently.  
All results are saved in the `./output/` directory.

---

## ⚙️ Environment Setup

All code runs on **Python ≥ 3.9** with GPU acceleration (recommended).

### 1️⃣ Install Dependencies

Run the following commands in your Kaggle or local terminal:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install lpips scikit-learn tqdm matplotlib pandas
```

### 2️⃣ Directory Structure

```
assignment3/
│
├── utils.py
├── train_baseline.py
├── adv_defense.py
├── poison_defense.py
├── mi_inv_defense.py
|── plot.py
|── README.md
|── output_new/
|── plots/
└── Report.pdf
```

The scripts automatically create:
```
/data/                ← CIFAR-10 dataset
/output/     ← CSVs + inversion image
/plots/           ← Generated result plots
```

---

## 🧩 Running the Code

### Option A: Run each defense separately

```bash
python train_baseline.py
python adv_defense.py
python poison_defense.py
python mi_inv_defense.py
```

### Option B: Run all sequentially

```bash
python train_baseline.py && python adv_defense.py && python poison_defense.py && python mi_inv_defense.py
```


### Option C: Generate all plots
After defenses are complete and CSVs are saved, run:
```bash
python plot.py
```

This will create six plots in the `plots/` directory.

---

## 📊 Output Summary

After execution, the `output/` folder will contain:

| File | Description |
|------|--------------|
| `train_history_baseline.csv` | Loss & accuracy per epoch for baseline training |
| `defense_adversarial.csv` | ε-sweep results (CleanAcc, AdvAcc, ASR, LPIPS) |
| `defense_poison.csv` | Fraction-sweep results (FilteredAcc, Removed) |
| `defense_mi.csv` | Membership inference comparison (Baseline vs Defended) |
| `defense_inversion.csv` | Gradient-regularization sweep (Lambda vs Accuracy) |
| `inversion_defended.png` | Final inversion image after defense |


## 📈 Example Plots (from `plot.py`)

| Plot | Description |
|------|-------------|
| **fgsm_asr.png** | Attack success rate vs ε |
| **fgsm_lpips.png** | LPIPS score vs ε |
| **fgsm_accuracy.png** | Clean vs Adversarial Accuracy vs ε |
| **poison_filtered_accuracy.png** | Accuracy vs fraction of filtered data |
| **mi_auc.png** | Membership inference comparison |
| **inversion_clean_accuracy.png** | Clean accuracy vs λ (gradient regularization) |

---

## 🧪 Expected Results (Latest Run)

| Defense | Key Metric | Observation |
|----------|-------------|-------------|
| FGSM Adv Training | ASR ↓ as ε ↑ | Trade-off between robustness & accuracy |
| Spectral Filtering | Best at 1–3% removal | Balances purity and accuracy |
| Label Smoothing + Temp | AUC = 0.537 → 0.669 | Better privacy, calibrated outputs |
| Gradient Regularization | λ=0.02 best | Stable accuracy, reduced inversion clarity |

Final baseline clean accuracy: **93.59%**

---