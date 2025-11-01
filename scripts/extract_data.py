import sys
import os
import torch
import soundfile as sf
import argparse
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import Generator
from scripts.embed_data import message_to_latent_vector # Reuse the same function

def compare_audio(audio_file_path: str, message: str, model: Generator, device: str, sample_rate: int, latent_dim: int, threshold=1e-5) -> bool:
    """
    Compares a given audio file with audio generated from a message to see if they match.

    Args:
        audio_file_path (str): Path to the audio file to check.
        message (str): The secret message to test for.
        model (Generator): The trained generator model.
        device (str): The device to run the model on ('cuda' or 'cpu').
        sample_rate (int): The sample rate of the audio.
        latent_dim (int): The dimensionality of the latent vector.
        threshold (float): The Mean Squared Error (MSE) threshold for a match.

    Returns:
        bool: True if the audio matches the message, False otherwise.
    """
    # 1. Load the received audio file
    try:
        received_audio, sr = sf.read(audio_file_path, dtype='float32')
        if sr != sample_rate:
            print(f"⚠️ Warning: Audio file sample rate ({sr}) differs from model sample rate ({sample_rate}).")
            # In a real-world scenario, you might want to resample here.
            # For now, we'll proceed but the comparison will likely fail.
        received_audio_tensor = torch.from_numpy(received_audio).to(device)
        print(f"🔍 Loaded audio file: {audio_file_path}")
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return False

    # 2. Generate the test audio from the message
    latent_vector = message_to_latent_vector(message, latent_dim).to(device)
    with torch.no_grad():
        test_audio_tensor = model(latent_vector).squeeze()

    # 3. Compare the two audio tensors
    if received_audio_tensor.shape != test_audio_tensor.shape:
        print("❌ Audio shapes do not match. Cannot compare.")
        return False

    mse = torch.mean((received_audio_tensor - test_audio_tensor) ** 2).item()
    print(f"📊 Calculated Mean Squared Error (MSE): {mse:.8f}")
    print(f"Threshold for match: {threshold:.8f}")

    return mse < threshold

def main():
    parser = argparse.ArgumentParser(description="Extract/verify a secret message from GAN-generated audio.")
    parser.add_argument("audio_file", type=str, help="Path to the .wav file to check.")
    parser.add_argument("message", type=str, help="The secret message to test for.")
    args = parser.parse_args()

    # --- Configuration ---
    LATENT_DIM = 100
    MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\generator_final.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_RATE = 16000
    AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1

    # --- Load Model ---
    print(f"💿 Loading trained generator model from {MODEL_PATH}")
    model = Generator(latent_dim=LATENT_DIM, output_length=AUDIO_LENGTH_SAMPLES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # --- Compare Audio ---
    is_match = compare_audio(args.audio_file, args.message, model, DEVICE, SAMPLE_RATE, LATENT_DIM)

    print("\n--- Result ---")
    print(f"✅ Message MATCH FOUND!" if is_match else "❌ Message does NOT match the audio.")

if __name__ == "__main__":
    main()