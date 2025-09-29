import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    A Discriminator model for classifying audio spectrograms as real or fake.

    The model is a Convolutional Neural Network (CNN) that takes a 2D spectrogram
    (with 2 channels for real and imaginary parts) and outputs a single
    probability score.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: (N, 2, F, T) where F is freq bins, T is time frames
            # We treat the real and imaginary parts of the STFT as 2 channels.
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 64, F/2, T/2)

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 128, F/4, T/4)

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, 256, F/8, T/8)

            # Final layer to produce a single output value
            # The kernel size might need adjustment based on the final feature map size
            # For now, we use a convolution that leads to a 1x1 output
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        # The input 'x' is the STFT of the audio, shape (N, F, T, 2)
        # We need to permute it to (N, 2, F, T) for Conv2d
        x = x.permute(0, 3, 1, 2)
        out = self.model(x)
        # Flatten the output to get a single score per item in the batch
        return out.view(-1, 1).squeeze(1)