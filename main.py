import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=1):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"Number of epochs = {epochs}")
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=1).permute(1, 2, 0))
    # plt.show()


latent_size = 128


def load_model(epochs):
    latent_size = 128
    generator = nn.Sequential(
        # in: latent_size x 1 x 1

        nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: 512 x 4 x 4

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: 256 x 8 x 8

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: 128 x 16 x 16

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: 64 x 32 x 32

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        # out: 3 x 64 x 64
    )
    generator.load_state_dict(torch.load(f"G{epochs}.pth", map_location=torch.device('cpu')))

    return generator
