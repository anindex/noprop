import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
import matplotlib.pyplot as plt
from noprop.model import NoPropFM
from sklearn.decomposition import PCA
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def visualize_vector_field_trajectories(model, data_loader, num_trajectories=8, steps=20):
    model.eval()
    x_batch, y_batch = next(iter(data_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    x_batch = x_batch[:num_trajectories]
    y_batch = y_batch[:num_trajectories]
    z1 = model.W_embed[y_batch]
    z0 = torch.randn_like(z1)

    traj_list = []
    for i in range(num_trajectories):
        z_t = []
        z = z0[i:i+1]
        u = z1[i:i+1]
        x = x_batch[i:i+1]

        for j in range(steps):
            t = torch.full((1, 1), j / steps, device=device)
            v = model.forward_vector_field(x, z, t)
            z_t.append(z.detach().cpu().numpy())
            z = model.extrapolate_z1(z, v, t)

        z_t.append(z.detach().cpu().numpy())
        traj = np.concatenate(z_t, axis=0)
        traj_list.append(traj)

    # Flatten all for PCA
    all_points = np.concatenate(traj_list, axis=0)
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_points)

    # Slice back into trajectories
    per_traj = steps + 1
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(num_trajectories):
        curve = z_2d[i * per_traj:(i + 1) * per_traj]
        ax.plot(curve[:, 0], curve[:, 1], marker='o', label=f'Class {y_batch[i].item()}')
        ax.text(curve[0, 0], curve[0, 1], "z₀", fontsize=8, color='gray')
        ax.text(curve[-1, 0], curve[-1, 1], "z₁", fontsize=8, color='black')

    ax.set_title("Vector Field Trajectories (PCA)")
    ax.legend()
    ax.grid(True)
    # plt.show()


def train_nopropfm(model, train_loader, test_loader, epochs, lr, weight_decay, inference_steps=1000):
    # NOTE: Adam is used in the paper (Table 3), but AdamW seems to work better in my case
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            B = x.size(0)

            z1 = model.W_embed[y]  # class embeddings
            z0 = torch.randn_like(z1)
            t = torch.rand(B, 1, device=device)

            z_t = t * z1 + (1 - t) * z0
            v_target = z1 - z0

            # Vector field prediction
            v_pred = model.forward_vector_field(x, z_t, t)
            loss_l2 = F.mse_loss(v_pred, v_target)

            # Extrapolate z̃₁ and compute classification loss
            z1_hat = model.extrapolate_z1(z_t, v_pred, t)
            logits = model.classifier(z1_hat)
            loss_ce = F.cross_entropy(logits, y)

            loss = loss_l2 + loss_ce

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

model = NoPropFM(num_classes=10, embedding_dim=embedding_dim).to(device)

train_nopropfm(model, train_loader, test_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, inference_steps=inference_steps)

class_names = [str(i) for i in range(10)]
num_images = 16

model.eval()
visualize_vector_field_trajectories(model, test_loader)

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
