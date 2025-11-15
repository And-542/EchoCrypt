import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    A Discriminator model for classifying raw audio waveforms as real or fake.
    This architecture mirrors the Extractor for consistency.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: (N, 1, 16000)
            nn.Conv1d(1, 32, kernel_size=5, stride=20, padding=2, bias=False), # -> (N, 32, 800)
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

            # Final layer to produce a single output score (1x1)
            nn.Conv1d(512, 1, kernel_size=4, stride=1, padding=0, bias=False), # -> (N, 1, 1)
        )

    def forward(self, x):
        # Input 'x' is the raw audio waveform, shape (N, 1, 16000)
        out = self.model(x)
        return out.view(-1) # Flatten to (N) for loss calculation