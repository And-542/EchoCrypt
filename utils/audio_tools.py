import torch

def get_stft(waveform: torch.Tensor, n_fft=400, hop_length=160, win_length=400) -> torch.Tensor:
    """
    Calculates the Short-Time Fourier Transform (STFT) of a waveform in a
    GPU-compatible way.

    Args:
        waveform (torch.Tensor): The input audio waveform. Shape: (batch_size, num_samples)

    Returns:
        torch.Tensor: A real-valued tensor representing the complex STFT.
                      Shape: (batch_size, freq_bins, time_frames, 2)
    """
    # Create a Hann window on the same device as the input waveform.
    window = torch.hann_window(window_length=win_length, device=waveform.device)

    # Calculate the complex-valued STFT. This operation is supported on CUDA.
    stft_complex = torch.stft(waveform, n_fft, hop_length, win_length, window, return_complex=True)

    # Convert the complex tensor to a real tensor with a new last dimension for real/imaginary parts.
    # This is the format the discriminator expects.
    return torch.view_as_real(stft_complex)