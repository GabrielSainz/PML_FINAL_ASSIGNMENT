# mnist_model_defs.py
# ============================================================
# Model definitions needed to load + sample your checkpoints
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Time embedding utilities
# -------------------------
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x should be shape (B,)
        if x.dim() != 1:
            x = x.view(-1)
        x_proj = x[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


# -------------------------
# Week 6 U-Net style ScoreNet
# -------------------------
class ScoreNet(nn.Module):
    """Time-dependent score model built upon U-Net architecture."""
    def __init__(self, marginal_prob_std, channels=(32, 64, 128, 256), embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoder
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoder
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

    def forward(self, x, t):
        # Ensure t is shape (B,)
        if t.dim() != 1:
            t = t.view(-1)

        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.conv1(x)
        h1 = h1 + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))

        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))

        h4 = self.conv4(h3)
        h4 = h4 + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        # Decoder
        h = self.tconv4(h4)
        h = h + self.dense5(embed)
        h = self.act(self.tgnorm4(h))

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = h + self.dense6(embed)
        h = self.act(self.tgnorm3(h))

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = h + self.dense7(embed)
        h = self.act(self.tgnorm2(h))

        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


# -------------------------
# EMA (not required for sampling, but harmless)
# -------------------------
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using exponential decay."""
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1.0 - decay) * model_param
        super().__init__(model, device, ema_avg, use_buffers=True)


# -------------------------
# Baseline DDPM wrapper (flat 784)
# -------------------------
class DDPM(nn.Module):
    def __init__(self, network, T=1000, beta_1=1e-4, beta_T=2e-2):
        super().__init__()
        self._network = network
        self.T = T

        # wrapper: (B,784) + (B,1 long) -> (B,784)
        self.network = lambda x, t: self._network(
            x.reshape(-1, 1, 28, 28),
            (t.squeeze() / T)
        ).reshape(-1, 28 * 28)

        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1.0 - self.alpha_bar[t])
        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        mean = (1.0 / torch.sqrt(self.alpha[t])) * (
            xt - (self.beta[t] / torch.sqrt(1.0 - self.alpha_bar[t])) * self.network(xt, t)
        )
        std = torch.where(
            t > 0,
            torch.sqrt(((1.0 - self.alpha_bar[t - 1]) / (1.0 - self.alpha_bar[t])) * self.beta[t]),
            torch.zeros_like(t, dtype=xt.dtype, device=xt.device),
        )
        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.beta.device)
        for step in range(self.T, 0, -1):
            t = torch.full((x.size(0), 1), step, device=x.device, dtype=torch.long)
            eps = torch.randn_like(x) if step > 1 else torch.zeros_like(x)
            x = self.reverse_diffusion(x, t, eps)
        return x

    def elbo_simple(self, x0):
        t = torch.randint(1, self.T, (x0.size(0), 1), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.forward_diffusion(x0, t, eps)
        return -F.mse_loss(eps, self.network(xt, t), reduction="mean")

    def loss(self, x0):
        return -self.elbo_simple(x0).mean()


# -------------------------
# Rectified Flow / Flow Matching
# -------------------------
class RectifiedFlow(nn.Module):
    """
    Linear interpolation:
      x_t = (1-t) x0 + t x1, x1 ~ N(0,I), t ~ U[0,1]
      target velocity: v* = x1 - x0
    """
    def __init__(self, unet_backbone):
        super().__init__()
        self._net = unet_backbone
        self.net = lambda x, t: self._net(
            x.view(-1, 1, 28, 28),
            t.view(-1)  # (B,)
        ).view(-1, 28 * 28)

    def loss(self, x0_flat):
        B = x0_flat.size(0)
        t = torch.rand(B, device=x0_flat.device)
        x1 = torch.randn_like(x0_flat)
        xt = (1.0 - t).unsqueeze(1) * x0_flat + t.unsqueeze(1) * x1
        v_target = x1 - x0_flat
        v_pred = self.net(xt, t)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, shape, steps=100, solver="heun"):
        """
        Integrate reverse-time ODE: t=1 -> t=0
          x_{t-dt} = x_t - v_theta(x_t, t) dt
        """
        B = shape[0]
        x = torch.randn(shape, device=next(self.parameters()).device)
        dt = 1.0 / steps

        if solver == "euler":
            for i in range(steps, 0, -1):
                t = torch.full((B,), i / steps, device=x.device)
                v = self.net(x, t)
                x = x - dt * v
            return x

        if solver == "heun":
            for i in range(steps, 0, -1):
                t1 = torch.full((B,), i / steps, device=x.device)
                v1 = self.net(x, t1)
                x_euler = x - dt * v1

                t0 = torch.full((B,), (i - 1) / steps, device=x.device)
                v0 = self.net(x_euler, t0)
                x = x - 0.5 * dt * (v1 + v0)
            return x

        raise ValueError("solver must be 'euler' or 'heun'")


# -------------------------
# Autoencoder (latent diffusion)
# -------------------------
class ConvAutoencoder(nn.Module):
    """
    28x28 -> 4x4 latent via strided convs
    Decoder maps back to 28x28.
    Output uses tanh => [-1,1]
    """
    def __init__(self, latent_c=16):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)         # 28->14
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)        # 14->7
        self.enc3 = nn.Conv2d(64, latent_c, 3, stride=2, padding=1)  # 7->4

        self.dec1 = nn.ConvTranspose2d(latent_c, 64, 3, stride=2, padding=1, output_padding=0)  # 4->7
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)       # 7->14
        self.dec3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)        # 14->28

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        z = self.enc3(x)
        return z

    def decode(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = torch.tanh(self.dec3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


# -------------------------
# Latent U-Net (padding=1) + DDPM2D
# -------------------------
def pick_num_groups(channels, max_groups=32):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ScoreNetPad(nn.Module):
    """Padding=1 U-Net variant for small spatial latents (e.g., 4x4)."""
    def __init__(self, marginal_prob_std, in_channels, channels=(64, 128, 256, 256), embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        c0, c1, c2, c3 = channels

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, c0, 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, c0)
        self.gn1 = nn.GroupNorm(pick_num_groups(c0), c0)

        self.conv2 = nn.Conv2d(c0, c1, 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, c1)
        self.gn2 = nn.GroupNorm(pick_num_groups(c1), c1)

        self.conv3 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, c2)
        self.gn3 = nn.GroupNorm(pick_num_groups(c2), c2)

        self.conv4 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, c3)
        self.gn4 = nn.GroupNorm(pick_num_groups(c3), c3)

        # Decoder
        self.tconv4 = nn.ConvTranspose2d(c3, c2, 3, stride=2, padding=1, output_padding=0, bias=False)
        self.dense5 = Dense(embed_dim, c2)
        self.tgn4 = nn.GroupNorm(pick_num_groups(c2), c2)

        self.tconv3 = nn.ConvTranspose2d(c2 + c2, c1, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, c1)
        self.tgn3 = nn.GroupNorm(pick_num_groups(c1), c1)

        self.tconv2 = nn.ConvTranspose2d(c1 + c1, c0, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, c0)
        self.tgn2 = nn.GroupNorm(pick_num_groups(c0), c0)

        self.tconv1 = nn.ConvTranspose2d(c0 + c0, in_channels, 3, stride=1, padding=1, output_padding=0)

    def forward(self, x, t):
        if t.dim() != 1:
            t = t.view(-1)
        embed = self.act(self.embed(t))

        h1 = self.act(self.gn1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gn2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gn3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gn4(self.conv4(h3) + self.dense4(embed)))

        h = self.act(self.tgn4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgn3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgn2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class DDPM2D(nn.Module):
    """
    DDPM wrapper for generic (C,H,W) tensors stored as flattened vectors.
    """
    def __init__(self, network_2d, C, H, W, T=1000, beta_1=1e-4, beta_T=2e-2):
        super().__init__()
        self._network = network_2d
        self.C, self.H, self.W = C, H, W
        self.D = C * H * W
        self.T = T

        # wrapper: (B, D) + (B,1 long) -> (B, D)
        self.network = lambda x, t: self._network(
            x.view(-1, C, H, W),
            (t.squeeze() / T)
        ).view(-1, self.D)

        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, eps):
        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1.0 - self.alpha_bar[t])
        return mean + std * eps

    def reverse_diffusion(self, xt, t, eps):
        mean = (1.0 / torch.sqrt(self.alpha[t])) * (
            xt - (self.beta[t] / torch.sqrt(1.0 - self.alpha_bar[t])) * self.network(xt, t)
        )
        std = torch.where(
            t > 0,
            torch.sqrt(((1.0 - self.alpha_bar[t - 1]) / (1.0 - self.alpha_bar[t])) * self.beta[t]),
            torch.zeros_like(t, dtype=xt.dtype, device=xt.device),
        )
        return mean + std * eps

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.beta.device)
        for step in range(self.T, 0, -1):
            t = torch.full((x.size(0), 1), step, device=x.device, dtype=torch.long)
            eps = torch.randn_like(x) if step > 1 else torch.zeros_like(x)
            x = self.reverse_diffusion(x, t, eps)
        return x

    def elbo_simple(self, x0):
        t = torch.randint(1, self.T, (x0.size(0), 1), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.forward_diffusion(x0, t, eps)
        return -F.mse_loss(eps, self.network(xt, t), reduction="mean")

    def loss(self, x0):
        return -self.elbo_simple(x0).mean()
