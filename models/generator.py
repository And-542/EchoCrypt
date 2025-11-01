import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    A Generator model for creating audio waveforms from a latent vector.

    The model uses a series of transposed convolutions to upsample a latent
    vector into a 1D audio signal.
    """

    def __init__(self, latent_dim=100, output_length=16000):
        """
        Initializes the Generator model.

        Args:
            latent_dim (int): The dimensionality of the input latent vector.
            output_length (int): The length of the output audio waveform.
                                 Note: The architecture may produce a slightly
                                 different length, which will be handled.
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length

        self.model = nn.Sequential(
            # Input: (N, latent_dim, 1)
            # Upsample latent vector to a small feature map
            nn.ConvTranspose1d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # State: (N, 512, 4)

            # Upsampling layers
            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # State: (N, 256, 8)

            nn.ConvTranspose1d(256, 128, 4, 4, 0, bias=False), # stride=4, padding=0 -> 128x32
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # State: (N, 128, 32)

            nn.ConvTranspose1d(128, 64, 5, 5, 0, bias=False), # stride=5, padding=0 -> 64x160
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # State: (N, 64, 160)

            # A single large kernel to generate the full waveform
            nn.ConvTranspose1d(64, 1, 100, 100, 0, bias=False), # -> (N, 1, 16000)
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, z):
        # z is the latent vector, shape (batch_size, latent_dim)
        # We need to reshape it to (batch_size, latent_dim, 1) for ConvTranspose1d
        out = self.model(z.unsqueeze(2))
        return out.squeeze(1)  # Return shape (batch_size, length)