import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
import matplotlib.pyplot as plt
from noprop.model import NoPropCT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_nopropct(model, train_loader, test_loader, epochs, lr, weight_decay, inference_steps=1000):
    # NOTE: Adam is used in the paper (Table 3), but AdamW seems to work better in my case
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            uy = model.W_embed[y]

            # Sample t ∈ [0, 1]
            t = torch.rand(B, 1, device=device)

            # Compute ᾱ(t)
            alpha_bar_t = model.alpha_bar(t)

            # Sample z_t ~ N(√ᾱ(t) * uy, (1 - ᾱ(t)) * I)
            noise = torch.randn_like(uy)
            z_t = torch.sqrt(alpha_bar_t) * uy + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict and compute loss
            z_pred = model.forward_denoise(x, z_t, t)
            snr_prime_t = model.snr_prime(t)
            loss_l2 = F.mse_loss(z_pred, uy)
            loss = 0.5 * model.eta * snr_prime_t.mean() * loss_l2

            # Final classifier loss (optional)
            logits = model.classifier(z_pred)
            loss_ce = F.cross_entropy(logits, y)
            loss_kl = 0.5 * uy.pow(2).sum(dim=1).mean()
            loss += loss_ce + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                preds = model.inference(x, steps=inference_steps).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            train_acc = correct / total

            correct, total = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model.inference(x, steps=inference_steps).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            val_acc = correct / total

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

eta = 1.0
embedding_dim = 512
batch_size = 128
lr = 1e-3
epochs = 10
weight_decay = 1e-3
inference_steps = 100  # inference_steps=1000 as Table 3 takes too long!


transform = transforms.ToTensor()

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader  = DataLoader(test_set,  batch_size = batch_size)

model = NoPropCT(num_classes=10, embedding_dim=embedding_dim, eta=eta).to(device)

train_nopropct(model, train_loader, test_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, inference_steps=inference_steps)

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
