import torch
import torch.nn as nn

class Extractor(nn.Module):
    """
    An Extractor model to recover a latent vector from an audio waveform.
    This architecture is the exact mirror of the new Generator.
    """
    def __init__(self, latent_dim=256):
        """
        Initializes the Extractor model.
        """
        super(Extractor, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Input: (N, 1, 16000)
            nn.Conv1d(1, 32, kernel_size=5, stride=20, padding=2, bias=False), # -> (N, 32, 800)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2, bias=False), # -> (N, 64, 200)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2, bias=False), # -> (N, 128, 40)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=5, stride=5, padding=2, bias=False), # -> (N, 256, 8)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), # -> (N, 512, 4)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(512, latent_dim, kernel_size=4, stride=1, padding=0, bias=False), # -> (N, latent_dim, 1)
            # No Tanh here, as the latent vector was not bounded to [-1, 1] before encoding
        )

    def forward(self, x):
        # x is the audio waveform, shape (batch_size, 1, length)
        out = self.model(x)
        # Squeeze the last two dimensions (H, W) which are (latent_dim, 1) -> (latent_dim)
        return out.squeeze(-1).squeeze(-1)