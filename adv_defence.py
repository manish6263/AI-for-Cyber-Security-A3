# adv_defense.py
"""
Adversarial defense script: FGSM adversarial training sweep.
Saves ./output_defenses/defense_adversarial.csv with rows for each epsilon.
"""

import torch, torch.nn.functional as F
import pandas as pd
from utils import create_resnet18, get_loaders, eval_acc, denorm, OUTDIR, DEVICE
import lpips
from tqdm import tqdm
import numpy as np

def fgsm(model, x, y, eps):
    x = x.clone().detach().to(DEVICE); x.requires_grad = True
    loss = F.cross_entropy(model(x), y.to(DEVICE))
    model.zero_grad(); loss.backward()
    adv = x + eps * x.grad.sign()
    return torch.clamp(adv, -5, 5)

def eval_adversarial_full(model, loader, eps):
    lp = lpips.LPIPS(net='alex').to(DEVICE)
    model.eval()
    c=a=flip=0
    total=0
    lp_vals=[]
    with torch.no_grad():
        for x,y in tqdm(loader, desc="FGSM Eval"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            pc = model(x).argmax(1)
            c += (pc == y).sum().item()
            adv = fgsm(model, x, y, eps)
            mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1).to(DEVICE)
            std  = torch.tensor([0.247,0.243,0.261]).view(1,3,1,1).to(DEVICE)
            adv_norm = (adv - mean) / std
            pa = model(adv_norm).argmax(1)
            a += (pa == y).sum().item()
            flip += ((pc==y) & (pa!=y)).sum().item()
            lp_vals.append(lp(denorm(x)*2-1, denorm(adv)*2-1).mean().item())
            total += y.size(0)
    return c/total, a/total, flip/max(1,c), np.mean(lp_vals)

def adversarial_defense(tr_loader, te_loader, eps_list=[4/255,8/255,12/255], epochs=10):
    results = []
    for eps in eps_list:
        print(f"\n--- Adversarial training with eps={eps:.5f} ---")
        model = create_resnet18().to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for e in range(1, epochs+1):
            model.train()
            for x,y in tr_loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                adv = fgsm(model, x, y, eps)
                opt.zero_grad()
                loss = F.cross_entropy(model(adv), y)
                loss.backward()
                opt.step()
            sched.step()
        acc_c, acc_a, asr, lp = eval_adversarial_full(model, te_loader, eps)
        print(f"eps={eps:.5f} -> Clean={acc_c:.3f}, Adv={acc_a:.3f}, ASR={asr:.3f}, LPIPS={lp:.6f}")
        results.append({"Epsilon": eps, "CleanAcc": acc_c, "AdvAcc": acc_a, "ASR": asr, "LPIPS": lp})
    pd.DataFrame(results).to_csv(f"{OUTDIR}/defense_adversarial.csv", index=False)
    print(f"Saved adversarial results to {OUTDIR}/defense_adversarial.csv")

if __name__ == "__main__":
    _, _, _, tr_loader, te_loader = get_loaders()
    adversarial_defense(tr_loader, te_loader)
