import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
import matplotlib.pyplot as plt
from noprop.model import NoPropDT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_nopropdt(model, train_loader, test_loader, epochs, lr, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()

        for t in tqdm(range(model.T)):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                uy = model.W_embed[y]
                alpha_bar_t = model.alpha_bar[t]
                noise = torch.randn_like(uy)
                z_t = torch.sqrt(alpha_bar_t) * uy + torch.sqrt(1 - alpha_bar_t) * noise

                z_pred, _ = model.blocks[t](x, z_t, model.W_embed)
                loss_l2 = F.mse_loss(z_pred, uy)
                loss = 0.5 * model.eta * model.snr_delta[t] * loss_l2

                if t == model.T - 1:
                    logits = model.classifier(z_pred)
                    loss_ce = F.cross_entropy(logits, y)
                    loss_kl = 0.5 * uy.pow(2).sum(dim=1).mean()
                    loss = loss + loss_ce + loss_kl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                preds = model.inference(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        train_acc = correct / total

        # Validation accuracy
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model.inference(x).argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total

        # Storing accuracy history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs}  "
              f"TrainAcc={100 * train_acc:.2f}%  ValAcc={100 * val_acc:.2f}%")

    # Plotting Training and Validation accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n Final Test Accuracy: {100 * val_acc:.2f}%")

T = 10
eta = 0.1
embedding_dim = 512
batch_size = 128
lr = 1e-3
epochs = 10
weight_decay = 1e-3


transform = transforms.ToTensor()

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader  = DataLoader(test_set,  batch_size = batch_size)

model = NoPropDT(num_classes=10, embedding_dim=embedding_dim, T=T, eta=eta).to(device)

train_nopropdt(model, train_loader, test_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)

class_names = [str(i) for i in range(10)]
num_images = 16

model.eval()
images_shown = 0
plt.figure(figsize=(5, 5))

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model.inference(x)
        preds = logits.argmax(dim=1)

        for i in range(x.size(0)):
            if images_shown >= num_images:
                break

            plt.subplot(int(num_images**0.5), int(num_images**0.5), images_shown + 1)
            img = x[i].cpu().squeeze(0)  
            plt.imshow(img, cmap='gray')
            actual = class_names[y[i]] if class_names else y[i].item()
            pred = class_names[preds[i]] if class_names else preds[i].item()
            plt.title(f"Pred: {pred}\nTrue: {actual}", fontsize=8)
            plt.axis('off')

            images_shown += 1

        if images_shown >= num_images:
            break

plt.tight_layout()
plt.show()
