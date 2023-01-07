from argparse import ArgumentParser
import functools
import code

import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import einops
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt


class VAE(pl.LightningModule):
    def __init__(self, latent_dim, data_module, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.data_module = data_module
        self.epoch_count = 0

    @staticmethod
    def add_model_specific_args(parser):
        parser = parser.add_argument_group("VAEModel")
        parser.add_argument("--latent_dim", type=int, default=10)
        return parser

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z, mu, log_var

    def loss(self, x_hat, x, mu, log_var):
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + kl

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, z, mu, log_var = self(x)
        loss = self.loss(x_hat, x, mu, log_var)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, z, mu, log_var = self(x)
        loss = self.loss(x_hat, x, mu, log_var)
        self.log("loss/val", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.epoch_count += 1
        self.logger.experiment.add_figure(
            "vae_output", self.draw_outputs(), self.epoch_count
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def draw_outputs(self):
        """Returns a matplotlib figure showing the VAE input and output."""
        plt.figure(figsize=(16, 4.5))
        images = self.data_module.template_images()
        n = 10
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            img = images[i].to(self.device)
            vae.eval()
            with torch.no_grad():
                x_hat, _, _, _ = vae(img)
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title("Original images")
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_hat.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title("Reconstructed images")
        return plt.gcf()


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.main_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0),  # 1x28x28 -> 32x12x12
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),  # 32x12x12 -> 64x4x4
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(64 * 4 * 4, 64 * 4 * 4),
            nn.ReLU(),
            nn.Linear(64 * 4 * 4, 64 * 4 * 4),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.LazyLinear(latent_dim),
        )
        self.log_var = nn.Sequential(
            nn.LazyLinear(latent_dim),
            nn.ReLU(),
        )

    def sample(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*std.size()).to(std)
        z = mu + std * esp
        return z

    def forward(self, x):
        x = self.main_block(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.sample(mu, log_var)
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.main_block = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Linear(64 * 4 * 4, 64 * 4 * 4),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c h w", c=64, h=4, w=4),
            nn.ConvTranspose2d(
                64, 32, 5, stride=2, output_padding=1
            ),  # 64x4x4 -> 32x12x12
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, 5, stride=2, output_padding=1
            ),  # 32x12x12 -> 1x28x28
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main_block(x)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir="data/", **kwargs):
        super().__init__()
        self.train_transform = self.test_transform = transforms.ToTensor()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def setup(self, stage=None):
        mnist_train = datasets.MNIST(
            self.data_dir,
            train=True,
            transform=self.train_transform,
            download=True,
        )
        mnist_test = datasets.MNIST(
            self.data_dir,
            train=False,
            transform=self.test_transform,
            download=True,
        )
        self.train_dataset = mnist_train
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )

    @functools.cache
    def template_images(self):
        """Returns a dict of image tensors that we use every epoch to judge the quality of the
        model."""
        labels = self.test_dataset.targets.numpy()
        d = {}
        n = 10
        for i in range(n):
            index = np.where(labels == i)[0][0]
            d[i] = einops.rearrange(self.test_dataset[index][0], "1 h w -> 1 1 h w")
        return d


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="data/")
    VAE.add_model_specific_args(parser)
    Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    mnist = MNISTDataModule(**dict_args)
    trainer = Trainer.from_argparse_args(args)
    vae = VAE(data_module=mnist, **dict_args)

    trainer.fit(vae, mnist)
