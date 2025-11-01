import sys
import os
import torch
import soundfile as sf
import argparse
import hashlib

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import Generator

def message_to_latent_vector(message: str, latent_dim: int) -> torch.Tensor:
    """
    Converts a string message into a deterministic latent vector.

    This function uses a cryptographic hash (SHA-256) of the message to seed a
    PyTorch random number generator. This ensures that the same message always
    produces the same latent vector, and that the vector's components are
    distributed similarly to the random noise the generator was trained on.

    Args:
        message (str): The secret message to embed.
        latent_dim (int): The dimensionality of the latent vector.

    Returns:
        torch.Tensor: A tensor of shape (1, latent_dim) to be used as input for the generator.
    """
    # 1. Create a hash of the message to get a fixed-size, deterministic seed
    hasher = hashlib.sha256(message.encode('utf-8'))
    seed_bytes = hasher.digest()

    # 2. Convert the first 8 bytes of the hash to a 64-bit integer seed
    seed = int.from_bytes(seed_bytes[:8], 'big')

    # 3. Seed the PyTorch random number generator
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 4. Generate the latent vector using the seeded generator
    # This creates a tensor with values from a standard normal distribution
    latent_vector = torch.randn(1, latent_dim, generator=generator)

    print(f"🌱 Message converted to latent vector with seed: {seed}")
    return latent_vector

def main():
    parser = argparse.ArgumentParser(description="Embed a secret message into GAN-generated audio.")
    parser.add_argument("message", type=str, help="The secret message to embed.")

    # Default output path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output = os.path.join(project_root, 'data', 'noise', 'embedded_message.wav')
    parser.add_argument("-o", "--output", type=str, default=default_output, help="Path to save the output .wav file.")
    args = parser.parse_args()

    # --- Configuration ---
    LATENT_DIM = 100
    MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\generator_final.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_RATE = 16000 # Must match training
    AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1 # Must match training

    # --- Load Model ---
    print(f"💿 Loading trained generator model from {MODEL_PATH}")
    model = Generator(latent_dim=LATENT_DIM, output_length=AUDIO_LENGTH_SAMPLES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # --- Generate Latent Vector from Message ---
    latent_vector = message_to_latent_vector(args.message, LATENT_DIM).to(DEVICE)

    # --- Generate Audio ---
    print("🎶 Generating audio from message...")
    with torch.no_grad():
        generated_waveform = model(latent_vector)

    # --- Save Audio ---
    audio_data = generated_waveform.squeeze().cpu().numpy()
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_data, samplerate=SAMPLE_RATE)
    print(f"✅ Audio with embedded message saved to {output_path}")

if __name__ == "__main__":
    main()
