import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the Generator and Discriminator networks
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define the DCGAN model
class DCGAN(nn.Module):
    def __init__(self, latent_dim, img_channels, dataset):
        super(DCGAN, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.generator = Generator(latent_dim, img_channels)
        self.discriminator = Discriminator(img_channels)

        self.batch_size = 128
        self.num_epochs = 10
        self.learning_rate = 0.0002

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        for epoch in range(self.num_epochs):
            for batch_idx, (real_images, _) in enumerate(self.dataloader):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Train Discriminator
                self.optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Generate fake images
                noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)

                # Forward pass through Generator
                fake_images = self.generator(noise)

                # Train Discriminator
                real_outputs = self.discriminator(real_images)
                fake_outputs = self.discriminator(fake_images.detach())

                d_loss_real = self.criterion(real_outputs, real_labels)
                d_loss_fake = self.criterion(fake_outputs, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer.step()

                # Train Generator
                self.optimizer.zero_grad()
                fake_outputs = self.discriminator(fake_images)
                g_loss = self.criterion(fake_outputs, real_labels)
                g_loss.backward()
                self.optimizer.step()

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.dataloader)}], "
                          f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")


if __name__ == '__main__':
    # Set the hyperparameters
    latent_dim = 100
    img_channels = 1

    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # Initialize the DCGAN model
    model = DCGAN(latent_dim, img_channels, mnist_dataset)

    # Train the DCGAN model
    model.train()
