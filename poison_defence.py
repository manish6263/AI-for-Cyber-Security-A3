import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import torch
from torch.utils.data import DataLoader, Subset
from utils import create_resnet18, get_loaders, eval_acc, OUTDIR, DEVICE

def spectral_filtering_and_retrain(base_model, trainset, test_loader, frac):
    # compute penultimate features using forward hook
    loader = DataLoader(trainset, batch_size=256, shuffle=False)
    feats_list = []
    labels_list = []
    hook_out = []
    def hook(_, __, out):
        hook_out.append(out.detach().cpu())
    h = base_model.avgpool.register_forward_hook(hook)
    base_model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            hook_out.clear(); _ = base_model(x)
            f = hook_out[0].view(x.size(0), -1)
            feats_list.append(f.cpu())
            labels_list.append(y)
    h.remove()
    feats = torch.cat(feats_list).numpy()
    labels = torch.cat(labels_list).numpy()
    susp = []
    for cls in np.unique(labels):
        idx = np.where(labels==cls)[0]
        f = feats[idx]
        mu = f.mean(0, keepdims=True)
        centered = f - mu
        sv = TruncatedSVD(n_components=1, random_state=0).fit(centered).components_[0]
        proj = np.abs(centered @ sv)
        k = int(len(idx) * frac)
        if k > 0:
            susp.extend(idx[np.argsort(-proj)[:k]])
    # retrain on filtered dataset
    keep = np.setdiff1d(np.arange(len(trainset)), susp)
    filt_loader = DataLoader(Subset(trainset, keep.tolist()), batch_size=128, shuffle=True, num_workers=2)
    model = create_resnet18().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    for e in range(10):
        model.train()
        for x,y in filt_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward(); opt.step()
    acc = eval_acc(model, test_loader)
    return acc, len(susp)

def poison_sweep(base_model, trainset, test_loader, fractions=[0.01,0.03,0.05]):
    results=[]
    for frac in fractions:
        print(f"\n--- Spectral filter fraction={frac} ---")
        acc, removed = spectral_filtering_and_retrain(base_model, trainset, test_loader, frac)
        print(f"Removed {removed} samples, filtered model acc = {acc:.4f}")
        results.append({"Fraction": frac, "FilteredAcc": acc, "Removed": removed})
    pd.DataFrame(results).to_csv(f"{OUTDIR}/defense_poison.csv", index=False)
    print(f"Saved poison defense CSV to {OUTDIR}/defense_poison.csv")

if __name__ == "__main__":
    # warm-up a base model (fast)
    trainset, testset, tr_loader, te_loader = get_loaders()
    base = create_resnet18().to(DEVICE)
    opt = torch.optim.SGD(base.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    print("Warming up base model for feature extraction (1 epoch)...")
    base.train()
    for x,y in tr_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); torch.nn.functional.cross_entropy(base(x), y).backward(); opt.step()
    poison_sweep(base, trainset, te_loader, fractions=[0.01,0.03,0.05])
