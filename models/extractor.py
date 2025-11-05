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
            nn.Conv1d(1, 64, kernel_size=10, stride=10, padding=0, bias=False), # -> (N, 64, 1600)
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 64, 1600)

            nn.Conv1d(64, 128, kernel_size=10, stride=10, padding=0, bias=False), # -> (N, 128, 160)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 128, 160)

            nn.Conv1d(128, 256, kernel_size=5, stride=5, padding=0, bias=False), # -> (N, 256, 32)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 256, 32)

            nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=0, bias=False), # -> (N, 512, 7)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 512, 7)

            # Flatten and project to latent dimension
            nn.Flatten(),
            nn.Linear(512 * 7, latent_dim),
            nn.Tanh() # Add Tanh to constrain output to [-1, 1]
        )

    def forward(self, x):
        # x is the audio waveform, shape (batch_size, length)
        # We need to reshape it to (batch_size, 1, length) for Conv1d
        out = self.model(x.unsqueeze(1))
        return out # Return shape (batch_size, latent_dim)