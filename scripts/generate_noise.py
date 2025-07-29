import numpy as np
import soundfile as sf

# Dummy waveform
noise = np.random.randn(16000)  # 1 sec at 16kHz
sf.write('data/noise/dummy_noise.wav', noise, samplerate=16000)
