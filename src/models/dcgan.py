import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class DCGAN:
    def __init__(self, dataset, nz=100, ngf=64, ndf=64, nc=3,
                 weights_save_path='./weights/dcgan_discriminator_weights.pth',
                 plots_save_dir='./outputs/plots'):
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.dataset = dataset
        self.weights_save_path = weights_save_path
        self.plots_save_dir = plots_save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.netG = self.Generator(nz, ngf, nc).to(self.device)
        self.netD = self.Discriminator(nc, ndf).to(self.device)

        self.losses = {'G': [], 'D': []}

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizerG = optim.SGD(self.netG.parameters(), lr=0.0002, momentum=0.9)
        self.optimizerD = optim.SGD(self.netD.parameters(), lr=0.0002, momentum=0.9)

    class Generator(nn.Module):
        def __init__(self, nz, ngf, nc):
            super(DCGAN.Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, x):
            return self.main(x)

    class Discriminator(nn.Module):
        def __init__(self, nc, ndf):
            super(DCGAN.Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            if isinstance(x, list):
                x = torch.stack(x)
            return self.main(x)

    def train(self, num_epochs=100):
        dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=2)
        dcgan_samples_dir = self.plots_save_dir.replace('plots', 'dcgan_samples')

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                data = data.to(self.device)
                batch_size = data.size(0)

                # Update Discriminator
                self.netD.zero_grad()
                label_real = torch.full((batch_size, 1), 1.0, device=self.device)
                output_real = self.netD(data).mean([2, 3])
                errD_real = self.criterion(output_real, label_real)
                errD_real.backward()

                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                fake_data = self.netG(noise)
                label_fake = torch.full((batch_size, 1), 0.0, device=self.device)
                output_fake = self.netD(fake_data.detach()).mean([2, 3])
                errD_fake = self.criterion(output_fake, label_fake)
                errD_fake.backward()
                self.optimizerD.step()

                # Update Generator
                self.netG.zero_grad()
                label_real.fill_(1.0)
                output = self.netD(fake_data).mean([2, 3])
                errG = self.criterion(output, label_real)
                errG.backward()
                self.optimizerG.step()

                self.losses['G'].append(errG.item())
                self.losses['D'].append(errD_real.item() + errD_fake.item())

                if i % 10 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD_real.item() + errD_fake.item(), errG.item()))

            # Save sample images per epoch
            os.makedirs(dcgan_samples_dir, exist_ok=True)
            fake = self.netG(torch.randn(64, self.nz, 1, 1, device=self.device))
            vutils.save_image(fake.detach(),
                              f'{dcgan_samples_dir}/fake_samples_epoch_{epoch + 1}.png',
                              normalize=True)

        # Save discriminator weights
        os.makedirs(os.path.dirname(self.weights_save_path), exist_ok=True)
        torch.save(self.netD.state_dict(), self.weights_save_path)

        # Plot training losses
        os.makedirs(self.plots_save_dir, exist_ok=True)
        plt.figure(figsize=(5, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.losses['G'], label="G")
        plt.plot(self.losses['D'], label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.ylim([-1, 2])
        plt.savefig(f'{self.plots_save_dir}/dcgan_loss.png')
        plt.clf()
