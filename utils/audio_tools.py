import torch

# STFT parameters
N_FFT = 1022
HOP_LENGTH = 256
WIN_LENGTH = 1022

def get_stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Computes the Short-Time Fourier Transform (STFT) of a waveform.

    Args:
        waveform (torch.Tensor): The input audio waveform. Shape: (batch, time).
        n_fft (int): Size of FFT.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): Each frame of audio is windowed by a window of this length.

    Returns:
        torch.Tensor: The complex-valued STFT. Shape: (batch, freq, time, 2 for real/imag).
    """
    window = torch.hann_window(win_length).to(waveform.device)
    stft_out = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False # Returns a real tensor of shape (batch, freq, time, 2)
    )
    return stft_out

def get_istft(stft_out, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Computes the inverse STFT to recover the audio waveform.

    Args:
        stft_out (torch.Tensor): The complex-valued STFT from get_stft.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): Each frame of audio is windowed by a window of this length.

    Returns:
        torch.Tensor: The reconstructed waveform.
    """
    # Convert the real tensor back to a complex tensor for istft
    stft_complex = torch.view_as_complex(stft_out)
    waveform = torch.istft(stft_complex, n_fft=N_FFT, hop_length=hop_length, win_length=win_length)
    return waveform