import torch
import numpy as np
from tqdm import tqdm
import sys
import os
import random
import string

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import Generator
from models.extractor import Extractor
from scripts.embed_data import message_to_binary_vector

# --- Configuration ---
NUM_TEST_MESSAGES = 10000  # Number of random messages to test
MAX_MESSAGE_LENGTH = 11    # Max characters per chunk (92 bits / 8 bits/char)

LATENT_DIM = 100
GEN_MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\generator_final.pth"
EXT_MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\extractor_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1

def generate_random_message(max_len: int) -> str:
    """Generates a random string of printable characters."""
    length = random.randint(1, max_len)
    return ''.join(random.choices(string.printable, k=length))

def main():
    """Main evaluation loop to calculate BER, MER, and SNR."""
    print("--- EchoCrypt Model Evaluation ---")
    print(f"Device: {DEVICE}")
    print(f"Testing with {NUM_TEST_MESSAGES} random messages.")
    print("----------------------------------")

    # --- Load Models ---
    print("💿 Loading Generator and Extractor models...")
    generator = Generator(latent_dim=LATENT_DIM, output_length=AUDIO_LENGTH_SAMPLES)
    generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    generator.to(DEVICE)
    generator.eval()

    extractor = Extractor(input_length=AUDIO_LENGTH_SAMPLES, latent_dim=LATENT_DIM)
    extractor.load_state_dict(torch.load(EXT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    extractor.to(DEVICE)
    extractor.eval()
    print("✅ Models loaded successfully.")

    # --- Initialize Metrics ---
    total_bits_processed = 0
    total_bit_errors = 0
    total_message_errors = 0
    snr_values = []

    # --- Evaluation Loop ---
    print("\n🚀 Starting evaluation...")
    loop = tqdm(range(NUM_TEST_MESSAGES), desc="Processing Messages")

    for _ in loop:
        # 1. Generate a random message and its corresponding vector
        original_message = generate_random_message(MAX_MESSAGE_LENGTH)
        try:
            # Note: message_to_binary_vector returns a batch of 1, so we squeeze it
            original_vector = message_to_binary_vector(original_message, LATENT_DIM).squeeze(0)
        except ValueError:
            continue # Should not happen with the length check

        # 2. Perform the encode-decode cycle
        with torch.no_grad():
            # Add batch dimension for models, move to device
            original_vector_batch = original_vector.unsqueeze(0).to(DEVICE)

            # Generate audio from the original vector
            generated_audio = generator(original_vector_batch)

            # Extract the vector from the generated audio
            extracted_vector_batch = extractor(generated_audio)

            # Remove batch dimension and move back to CPU for analysis
            extracted_vector = extracted_vector_batch.squeeze(0).cpu()

        # 3. Calculate metrics for this sample
        # Convert original vector from {-1, 0, 1} to binary {0, 1} for comparison
        # We only care about the data bits, which are -1 or 1. Padding (0) is ignored.
        data_indices = torch.where(original_vector != 0)[0]
        original_bits = (original_vector[data_indices] > 0).int() # 1 for 1.0, 0 for -1.0

        # Threshold the extracted vector to get binary bits
        reconstructed_bits = (extracted_vector[data_indices] > 0).int()

        # --- Bit Error Rate (BER) ---
        bit_errors = torch.sum(original_bits != reconstructed_bits).item()
        total_bit_errors += bit_errors
        total_bits_processed += len(data_indices)

        # --- Message Error Rate (MER) ---
        if bit_errors > 0:
            total_message_errors += 1

        # --- Signal-to-Noise Ratio (SNR) ---
        # Use the raw float vectors for SNR calculation
        signal_power = torch.mean(original_vector[data_indices] ** 2).item()
        noise_power = torch.mean((original_vector[data_indices] - extracted_vector[data_indices]) ** 2).item()

        if noise_power > 0 and signal_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            snr_values.append(snr)

    print("✅ Evaluation complete.")

    # --- Final Results ---
    ber = (total_bit_errors / total_bits_processed) if total_bits_processed > 0 else 0
    mer = (total_message_errors / NUM_TEST_MESSAGES) if NUM_TEST_MESSAGES > 0 else 0
    avg_snr = np.mean(snr_values) if snr_values else float('nan')

    print("\n--- Final Metrics ---")
    print(f"Total Messages Tested: {NUM_TEST_MESSAGES}")
    print(f"Total Bits Processed:  {total_bits_processed}")
    print("-----------------------")
    print(f"📊 Bit Error Rate (BER):    {ber:.8f} ({total_bit_errors} errors)")
    print(f"📊 Message Error Rate (MER):  {mer:.4f} ({total_message_errors} errors)")
    print(f"📈 Average SNR (dB):        {avg_snr:.2f} dB")
    print("-----------------------")


if __name__ == "__main__":
    main()