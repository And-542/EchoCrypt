import torch
import torch.nn as nn

class Extractor(nn.Module):
    """
    An Extractor model to recover a latent vector from an audio waveform.

    In the context of steganography, this model acts as the **Decoder**, as it
    extracts the hidden information (the latent vector) from the carrier signal (the audio).

    In the context of an autoencoder architecture, this model acts as the **Encoder**,
    as it compresses the high-dimensional audio input into a low-dimensional latent space.
    It is designed to be the architectural inverse of the Generator (which acts as the Decoder
    in an autoencoder).
    """
    def __init__(self, input_length=16000, latent_dim=100):
        """
        Initializes the Extractor model.

        Args:
            input_length (int): The length of the input audio waveform.
            latent_dim (int): The dimensionality of the output latent vector.
        """
        super(Extractor, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Input: (N, 1, 16000)
            nn.Conv1d(1, 64, kernel_size=500, stride=500, padding=0, bias=False), # -> (N, 64, 32)
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 64, 32)

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 128, 16)

            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 256, 8)

            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 512, 4)

            # Flatten and project to latent dimension
            nn.Conv1d(512, latent_dim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Tanh(), # Add Tanh to constrain output to [-1, 1]
            nn.Flatten() # Flatten to (N, latent_dim)
        )

    def forward(self, x):
        # x is the audio waveform, shape (batch_size, length)
        # We need to reshape it to (batch_size, 1, length) for Conv1d
        out = self.model(x.unsqueeze(1))
        return out.squeeze(-1) # Squeeze the last dimension to get (N, latent_dim)