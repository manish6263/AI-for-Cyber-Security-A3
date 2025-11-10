import matplotlib.pyplot as plt
import pandas as pd
import os

PLOT_DIR = "plots_new"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# FGSM
adv = pd.read_csv("./output_new/defense_adversarial.csv")

plt.figure()
plt.plot(adv["Epsilon"], adv["ASR"], marker='o')
plt.xlabel("Epsilon")
plt.ylabel("ASR")
plt.title("ASR vs Epsilon")
plt.savefig(f"{PLOT_DIR}/fgsm_asr.png")

plt.figure()
plt.plot(adv["Epsilon"], adv["LPIPS"], marker='o')
plt.xlabel("Epsilon")
plt.ylabel("LPIPS")
plt.title("LPIPS vs Epsilon")
plt.savefig(f"{PLOT_DIR}/fgsm_lpips.png")

plt.figure()
plt.plot(adv["Epsilon"], adv["CleanAcc"], 'g-o', label='Clean')
plt.plot(adv["Epsilon"], adv["AdvAcc"], 'r-o', label='Adv')
plt.legend()
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon")
plt.savefig(f"{PLOT_DIR}/fgsm_accuracy.png")

# Poisoning
poi = pd.read_csv("./output_new/defense_poison.csv")

plt.figure()
plt.plot(poi["Fraction"], poi["FilteredAcc"], 'b-o')
plt.xlabel("Fraction Removed")
plt.ylabel("Accuracy")
plt.title("Filtered Accuracy vs Fraction")
plt.savefig(f"{PLOT_DIR}/poison_filtered_accuracy.png")

# Membership Inference
mi = pd.read_csv("./output_new/defense_mi.csv")
plt.figure()
plt.bar(mi["Model"], mi["AUC"], color=['gray','green'])
plt.ylabel("AUC")
plt.title("Membership Inference AUC")
plt.savefig(f"{PLOT_DIR}/mi_auc.png")

# Model Inversion
inv = pd.read_csv("./output_new/defense_inversion.csv")
plt.figure()
plt.plot(inv["Lambda"], inv["CleanAcc"], 'm-o')
plt.xlabel("Lambda")
plt.ylabel("Clean Accuracy")
plt.title("Clean Accuracy vs Lambda (Gradient Regularization)")
plt.savefig(f"{PLOT_DIR}/inversion_clean_accuracy.png")