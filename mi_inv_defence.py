# mi_inv_defense.py
"""
Membership inference defense (baseline vs label-smoothing + temp scaling)
and model inversion defense (gradient regularization sweep).
Saves defense_mi.csv, defense_inversion.csv, and inversion_defended.png
"""

import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import create_resnet18, get_loaders, eval_acc, denorm, OUTDIR, DEVICE
from sklearn.metrics import roc_auc_score

def compute_mi_auc(model, trainset, testset, temp=1.5):
    def confs(ds):
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        out = []
        with torch.no_grad():
            for x,y in loader:
                p = F.softmax(model(x.to(DEVICE))/temp, 1)
                out.append(p.max(1)[0].cpu().numpy())
        return np.concatenate(out)
    tr_conf = confs(trainset); te_conf = confs(testset)
    y_true = np.concatenate([np.ones(len(tr_conf)), np.zeros(len(te_conf))])
    y_score = np.concatenate([tr_conf, te_conf])
    return roc_auc_score(y_true, y_score)

def mi_defense(tr_loader, te_loader, trainset, testset):
    results = []
    # baseline small training
    base = create_resnet18().to(DEVICE)
    opt = torch.optim.SGD(base.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    for e in range(5):
        base.train()
        for x,y in tr_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(base(x), y).backward(); opt.step()
    auc_base = compute_mi_auc(base, trainset, testset)
    print(f"Baseline MI AUC = {auc_base:.3f}")
    results.append({"Model": "Baseline", "AUC": auc_base})

    # defended model (label smoothing)
    m = create_resnet18().to(DEVICE)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    smooth = 0.1
    for e in range(1,31):
        m.train()
        for x,y in tr_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = m(x); logp = F.log_softmax(logits, 1)
            k = logits.size(1)
            sm = torch.full((y.size(0),k), smooth/(k-1)).to(DEVICE)
            sm.scatter_(1, y.unsqueeze(1), 1-smooth)
            loss = F.kl_div(logp, sm, reduction='batchmean')
            loss.backward(); opt.step()
        sch.step()
    auc_def = compute_mi_auc(m, trainset, testset)
    print(f"Defended MI AUC = {auc_def:.3f}")
    results.append({"Model": "Defended", "AUC": auc_def})
    pd.DataFrame(results).to_csv(f"{OUTDIR}/defense_mi.csv", index=False)
    print(f"Saved MI results to {OUTDIR}/defense_mi.csv")

def model_inversion_defense(tr_loader, te_loader, lambda_list=[0.0,0.02,0.05,0.1]):
    results = []
    last_model = None
    for lam in lambda_list:
        print(f"\n--- Training with lambda={lam} ---")
        m = create_resnet18().to(DEVICE)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        for e in range(1,21):
            m.train()
            for x,y in tr_loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                x.requires_grad = True
                logits = m(x); loss = F.cross_entropy(logits, y)
                loss.backward(retain_graph=True)
                if x.grad is not None:
                    grad_pen = (x.grad.view(x.size(0),-1).norm(2,1)**2).mean()
                else:
                    grad_pen = torch.tensor(0.).to(DEVICE)
                opt.zero_grad()
                (loss + lam * grad_pen).backward()
                opt.step()
        acc = eval_acc(m, te_loader)
        print(f"lambda={lam:.3f} -> Clean acc={acc:.4f}")
        results.append({"Lambda": lam, "CleanAcc": acc})
        last_model = m
    # create inversion sample for the last model (strongest defense)
    x = torch.rand(1,3,32,32,device=DEVICE,requires_grad=True)
    opt_x = torch.optim.Adam([x], lr=0.05)
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1).to(DEVICE)
    std  = torch.tensor([0.247,0.243,0.261]).view(1,3,1,1).to(DEVICE)
    for i in range(300):
        opt_x.zero_grad()
        logits = last_model((x-mean)/std)
        loss = -logits[0,0]
        loss.backward()
        opt_x.step()
        with torch.no_grad():
            x.clamp_(0,1)
    save_image(denorm(x), f"{OUTDIR}/inversion_defended.png")
    pd.DataFrame(results).to_csv(f"{OUTDIR}/defense_inversion.csv", index=False)
    print(f"Saved inversion image to {OUTDIR}/inversion_defended.png and CSV to {OUTDIR}/defense_inversion.csv")

if __name__ == "__main__":
    trainset, testset, tr_loader, te_loader = get_loaders()
    mi_defense(tr_loader, te_loader, trainset, testset)
    model_inversion_defense(tr_loader, te_loader, lambda_list=[0.0,0.02,0.05,0.1])
