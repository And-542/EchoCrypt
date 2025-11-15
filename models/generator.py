import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    A Generator model for creating audio waveforms from a latent vector.

    The model uses a series of transposed convolutions to upsample a latent
    vector into a 1D audio signal.
    """

    def __init__(self, latent_dim=256): # output_length is implicitly 16000
        """
        Initializes the Generator model.

        Args:
            latent_dim (int): The dimensionality of the input latent vector.
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Input: (N, latent_dim, 1)
            nn.ConvTranspose1d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False), # -> (N, 512, 4)
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), # -> (N, 256, 8)
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=5, padding=2, output_padding=4, bias=False), # -> (N, 128, 40)
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=5, padding=2, output_padding=4, bias=False), # -> (N, 64, 200)
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=4, padding=2, output_padding=3, bias=False), # -> (N, 32, 800)
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=20, padding=2, output_padding=19, bias=False), # -> (N, 1, 16000)
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, z):
        # z is the latent vector, shape (batch_size, latent_dim)
        # We need to reshape it to (batch_size, latent_dim, 1) for ConvTranspose1d
        out = self.model(z.unsqueeze(2)) # Add a dummy dimension for ConvTranspose1d
        return out # Output shape: (N, 1, 16000)