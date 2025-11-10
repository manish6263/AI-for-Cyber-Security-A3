import torch, torch.nn.functional as F
import pandas as pd
from utils import get_loaders, create_resnet18, eval_acc, OUTDIR, DEVICE

def train_baseline(epochs=30, lr=0.1):
    trainset,testset,train_loader,test_loader = get_loaders()
    model = create_resnet18().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    print("Training baseline ResNet-18")
    for e in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item(); batches += 1
        sched.step()
        if e % 5 == 0 or e == epochs:
            acc = eval_acc(model, test_loader)
            print(f"Epoch {e}/{epochs}: acc={acc:.4f}")
        history.append({"epoch": e, "train_loss": epoch_loss/batches})
    acc_final = eval_acc(model, test_loader)
    torch.save(model.state_dict(), f"{OUTDIR}/baseline.pth")
    print(f"Baseline done. Clean acc={acc_final:.4f}")
    pd.DataFrame(history).to_csv(f"{OUTDIR}/train_history_baseline.csv", index=False)
    return model, trainset, testset, train_loader, test_loader

if __name__ == "__main__":
    train_baseline()