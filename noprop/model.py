import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import DenoiseBlock


class NoPropDT(nn.Module):
    def __init__(self, num_classes, embedding_dim, T, eta):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        self.blocks = nn.ModuleList([
            DenoiseBlock(embedding_dim, num_classes) for _ in range(T)
        ])

        # Learnable class embeddings
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)

        # Output classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Cosine noise schedule
        t = torch.arange(1, T+1, dtype=torch.float32)
        alpha_t = torch.square(torch.cos(t / T * (torch.pi / 2)))
        alpha_bar = torch.cumprod(alpha_t, dim=0)
        snr = alpha_bar / (1 - alpha_bar)
        snr_prev = torch.cat([torch.tensor([0.], dtype=snr.dtype), snr[:-1]], dim=0)
        snr_delta = snr - snr_prev

        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('snr_delta', snr_delta)

    def forward_denoise(self, x, z_prev, t):
        return self.blocks[t](x, z_prev, self.W_embed)[0]

    def classify(self, z):
        return self.classifier(z)

    def inference(self, x):
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)

        for t in range(self.T):
            z = self.forward_denoise(x, z, t)

        return self.classify(z)


class NoPropCT(nn.Module):
    def __init__(self, num_classes, embedding_dim, eta):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.eta = eta

        # Learnable class embeddings
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)

        # Output classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Positional embedding for time t
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Input fusion block û_θ(z, x, t)
        self.image_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.latent_embed = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU()
        )

        self.u_theta = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

        # Learnable γ(t) for noise schedule: ᾱ(t) = σ(−γ(t))
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.gamma_0 = nn.Parameter(torch.tensor(1.0))
        self.gamma_1 = nn.Parameter(torch.tensor(5.0))

    def alpha_bar(self, t):
        gamma_hat = self.gamma_mlp(t)  # Ensure positivity
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (1 - gamma_hat / gamma_hat.max())
        return torch.sigmoid(-gamma)

    def snr_prime(self, t):
        # d/dt [ᾱ / (1 - ᾱ)] = ᾱ'(1 - ᾱ + ᾱ) / (1 - ᾱ)^2 = ᾱ' / (1 - ᾱ)^2
        t.requires_grad_(True)
        alpha = self.alpha_bar(t)
        grad = torch.autograd.grad(alpha.sum(), t, create_graph=True)[0]
        snr_prime = grad / (1 - alpha) ** 2
        return snr_prime

    def forward_denoise(self, x, z_t, t):
        x_feat = self.image_embed(x)         # [B, 128]
        z_feat = self.latent_embed(z_t)      # [B, 128]
        t_feat = self.time_embed(t)          # [B, 64]

        # Now all are [B, D] so we can safely concatenate
        fused = torch.cat([x_feat, z_feat, t_feat], dim=1)
        weights = self.u_theta(fused)
        return weights @ self.W_embed

    def inference(self, x, steps=1000):
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)

        for i in range(steps):
            t = torch.full((B, 1), i / steps, device=x.device)
            z = self.forward_denoise(x, z, t)

        return self.classifier(z)


class NoPropFM(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Learnable class embeddings
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)

        # Positional encoding of time
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Image embedding pathway
        self.image_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Latent z embedding
        self.latent_embed = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU()
        )

        # Vector field network vθ(z, x, t)
        self.vector_field = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Output classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_vector_field(self, x, z_t, t):
        B = x.size(0)
        x_feat = self.image_embed(x)
        z_feat = self.latent_embed(z_t)
        t_feat = self.time_embed(t)  # shape [B, 64]
        fused = torch.cat([x_feat, z_feat, t_feat], dim=1)
        return self.vector_field(fused)  # [B, D]

    def extrapolate_z1(self, z_t, v_t, t):
        return z_t + (1 - t) * v_t  # z̃₁ = zₜ + (1 - t) * vθ(zₜ, x, t)

    def inference(self, x, steps=1000):
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)

        for i in range(steps):
            t = torch.full((B, 1), i / steps, device=x.device)
            v_t = self.forward_vector_field(x, z, t)
            z = self.extrapolate_z1(z, v_t, t)

        return self.classifier(z)
