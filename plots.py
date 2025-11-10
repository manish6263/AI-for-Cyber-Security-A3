import matplotlib.pyplot as plt
import pandas as pd
import os

# FGSM
PLOT_DIR = "plots_new"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
adv = pd.read_csv("./output_new/defense_adversarial.csv")
plt.figure(); plt.plot(adv["Epsilon"], adv["ASR"], marker='o'); plt.xlabel("Epsilon"); plt.ylabel("ASR"); plt.title("ASR vs Epsilon"); plt.savefig(f"{PLOT_DIR}/fgsm_asr.png")
plt.figure(); plt.plot(adv["Epsilon"], adv["LPIPS"], marker='o'); plt.xlabel("Epsilon"); plt.ylabel("LPIPS"); plt.title("LPIPS vs Epsilon"); plt.savefig(f"{PLOT_DIR}/fgsm_lpips.png")
plt.figure(); plt.plot(adv["Epsilon"], adv["CleanAcc"], 'g-o', label='Clean'); plt.plot(adv["Epsilon"], adv["AdvAcc"], 'r-o', label='Adv'); plt.legend(); plt.xlabel("Epsilon"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epsilon"); plt.savefig(f"{PLOT_DIR}/fgsm_accuracy.png")

# Poisoning
poi = pd.read_csv("./output_new/defense_poison.csv")
plt.figure(); plt.plot(poi["Fraction"], poi["FilteredAcc"], 'b-o'); plt.xlabel("Fraction Removed"); plt.ylabel("Accuracy"); plt.title("Filtered Accuracy vs Fraction"); plt.savefig(f"{PLOT_DIR}/poison_filtered_accuracy.png")

# Membership Inference
mi = pd.read_csv("./output_new/defense_mi.csv")
plt.figure(); plt.bar(mi["Model"], mi["AUC"], color=['gray','green']); plt.ylabel("AUC"); plt.title("Membership Inference AUC"); plt.savefig(f"{PLOT_DIR}/mi_auc.png")

# Model Inversion
inv = pd.read_csv("./output_new/defense_inversion.csv")
plt.figure(); plt.plot(inv["Lambda"], inv["CleanAcc"], 'm-o'); plt.xlabel("Lambda"); plt.ylabel("Clean Accuracy"); plt.title("Clean Accuracy vs Lambda (Gradient Regularization)"); plt.savefig(f"{PLOT_DIR}/inversion_clean_accuracy.png")













# 🚀 Training baseline ResNet-18...
# 100%|██████████| 170M/170M [00:09<00:00, 17.4MB/s] 
# Epoch 5/30: acc=0.7086
# Epoch 10/30: acc=0.7618
# Epoch 15/30: acc=0.8725
# Epoch 20/30: acc=0.8866
# Epoch 25/30: acc=0.9258
# Epoch 30/30: acc=0.9359
# ✅ Baseline done. Clean acc=0.9359

# 🛡️ Adversarial Defense (FGSM Adv Training)

# --- Training with ε=0.01569 ---
# Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
# /usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
# 100%|██████████| 233M/233M [00:01<00:00, 181MB/s]  
# Loading model from: /usr/local/lib/python3.11/dist-packages/lpips/weights/v0.1/alex.pth
# ε=0.01569 → Clean=0.866, Adv=0.224, ASR=0.760, LPIPS=0.0001

# --- Training with ε=0.03137 ---
# Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
# Loading model from: /usr/local/lib/python3.11/dist-packages/lpips/weights/v0.1/alex.pth
# ε=0.03137 → Clean=0.840, Adv=0.245, ASR=0.732, LPIPS=0.0005

# --- Training with ε=0.04706 ---
# Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
# Loading model from: /usr/local/lib/python3.11/dist-packages/lpips/weights/v0.1/alex.pth
# ε=0.04706 → Clean=0.815, Adv=0.261, ASR=0.715, LPIPS=0.0012

# 🧩 Poisoning Defense (Spectral Signature Filtering)

# --- Fraction=0.01 ---
# Detected 500 suspicious samples.
# Fraction=0.01 → Acc=0.7933

# --- Fraction=0.03 ---
# Detected 1500 suspicious samples.
# Fraction=0.03 → Acc=0.7855

# --- Fraction=0.05 ---
# Detected 2500 suspicious samples.
# Fraction=0.05 → Acc=0.7461

# 🔒 Membership Inference Defense (Label Smoothing + Temp Scaling)
# Baseline MI AUC=0.537
# Defended MI AUC=0.669

# 🧠 Model Inversion Defense (Gradient Regularization)

# --- λ=0.0 ---
# λ=0.00 → Clean acc=0.7896

# --- λ=0.02 ---
# λ=0.02 → Clean acc=0.8254

# --- λ=0.05 ---
# λ=0.05 → Clean acc=0.7909

# --- λ=0.1 ---
# λ=0.10 → Clean acc=0.7534
# ✅ Saved inversion_defended.png

# 🎯 All parameter sweeps complete. Check ./output_defenses for CSVs and images.