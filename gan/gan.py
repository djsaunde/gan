import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False

os.makedirs('../images', exist_ok=True)


class Generator(nn.Module):
    # language=rst
    """
    Generator network of the generative adversarial network.
    """

    def __init__(self, image_shape, n_latent):
        # language=rst
        """
        Constructor of the generator network.

        :param image_shape: Dimensionality of the input images.
        :param n_latent: Number of neuron in latent vector space.
        """
        super(Generator, self).__init__()

        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(n_latent, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(image_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        # language=rst
        """
        Forward pass of the generator network.

        :param z: Sample(s) from the latent vector space.
        """
        image = self.model(z)

        image = image.view(image.size(0), *self.image_shape)
        return image


class Discriminator(nn.Module):
    # language=rst
    """
    Discriminator network of the generative adversarial network.
    """

    def __init__(self, image_shape):
        # language=rst
        """
        Constructor for discriminator network.
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        validity = self.model(flat)
        return validity


def main(image_size=(28, 28), channels=1, n_latent=50, batch_size=50, n_epochs=25, sample_interval=400):
    image_shape = (channels, image_size[0], image_size[1])

    # Initialize generator and discriminator networks.
    generator = Generator(image_shape, n_latent)
    discriminator = Discriminator(image_shape)

    # Initialize adversarial loss.
    loss = nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()
    
    # Configure data loader
    os.makedirs('../../data/mnist', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../../data/mnist', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ), batch_size=batch_size, shuffle=True
    )

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(n_epochs):
        for i, (images, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Tensor(images.size(0), 1).fill_(1.0)
            fake = Tensor(images.size(0), 1).fill_(0.0)

            # Configure input
            real_imgs = images.type(Tensor)

            # -----------------
            #  Train Generator
            # -----------------

            generator_optimizer.zero_grad()

            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (images.shape[0], n_latent)))

            # Generate a batch of images
            generated = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = loss(discriminator(generated), valid)
            g_loss.backward()
            generator_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            discriminator_optimizer.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(discriminator(real_imgs), valid)
            fake_loss = loss(discriminator(generated.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            discriminator_optimizer.step()

            print(
                f'[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] '
                f'[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]'
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(generated.data[:25], f'../images/{batches_done}.png', nrow=5, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs=2, default=[28, 28])
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--n_latent', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=400)
    args = parser.parse_args()

    image_size = args.image_size
    channels = args.channels
    n_latent = args.n_latent
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    sample_interval = args.sample_interval

    main(
        image_size=image_size, channels=channels, n_latent=n_latent,
        batch_size=batch_size, n_epochs=n_epochs, sample_interval=sample_interval
    )
