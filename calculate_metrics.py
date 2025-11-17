import torch
import numpy as np
from tqdm import tqdm
import sys
import os
import random
import string

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reedsolo import RSCodec
from models.generator import Generator
from models.extractor import Extractor
from utils.conversion import data_to_binary_vector, binary_vector_to_data

# --- Configuration ---
NUM_TEST_MESSAGES = 10000  # Number of random messages to test
LATENT_DIM = 256

# --- FEC Configuration (must match app.py) ---
FEC_SYMBOLS = 16
MAX_CHUNK_SIZE = (LATENT_DIM - 8) // 8
MAX_PAYLOAD_BYTES_PER_CHUNK = MAX_CHUNK_SIZE - FEC_SYMBOLS
MAX_MESSAGE_LENGTH = MAX_PAYLOAD_BYTES_PER_CHUNK * 4 # Test with messages up to 4 chunks long

GEN_MODEL_PATH = "d:/EchoCrypt/EchoCrypt/models/saved_models/generator_final.pth"
EXT_MODEL_PATH = "d:/EchoCrypt/EchoCrypt/models/saved_models/extractor_final.pth"
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
    generator = Generator(latent_dim=LATENT_DIM)
    generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    generator.to(DEVICE)
    generator.eval()

    extractor = Extractor(latent_dim=LATENT_DIM)
    extractor.load_state_dict(torch.load(EXT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    extractor.to(DEVICE)
    extractor.eval()
    print("✅ Models loaded successfully.")

    # --- Initialize Metrics ---
    total_message_errors = 0
    total_bit_errors = 0
    total_bits_tested = 0
    total_snr = 0.0
    total_chunks_processed = 0
    rs = RSCodec(FEC_SYMBOLS)

    # --- Evaluation Loop ---
    print("\n🚀 Starting evaluation...")
    loop = tqdm(range(NUM_TEST_MESSAGES), desc="Processing Messages")

    for _ in loop:
        # 1. Generate a random message and chunk it
        original_message_str = generate_random_message(MAX_MESSAGE_LENGTH)
        original_message_bytes = original_message_str.encode('utf-8')

        # This is a dummy encryption step to simulate the app's payload structure.
        # We don't need a password since we are comparing the final bytes directly.
        # The key is that the payload is chunked AFTER processing.
        payload_to_hide = original_message_bytes # In the app, this would be salt+iv+ciphertext

        payload_chunks = [payload_to_hide[i:i + MAX_PAYLOAD_BYTES_PER_CHUNK] for i in range(0, len(payload_to_hide), MAX_PAYLOAD_BYTES_PER_CHUNK)]

        reconstructed_payload = b""
        is_corrupted = False

        # 2. Process each chunk through the full autoencoder and FEC pipeline
        for chunk in payload_chunks:
            try:
                # Add FEC
                fec_chunk = rs.encode(chunk)

                # Convert to vector
                original_vector = data_to_binary_vector(fec_chunk, LATENT_DIM).to(DEVICE)

                # Autoencoder pass
                with torch.no_grad():
                    generated_audio = generator(original_vector)
                    extracted_vector = extractor(generated_audio)

                # --- Calculate BER (before FEC) ---
                original_vec_cpu = original_vector.squeeze().cpu().numpy()
                extracted_vec_cpu = extracted_vector.squeeze().cpu().numpy()
                binarized_extracted = np.where(extracted_vec_cpu > 0, 1.0, -1.0)
                
                # Only compare non-padded bits
                data_indices = np.where(original_vec_cpu != 0.0)[0]
                if len(data_indices) > 0:
                    bit_errors_in_chunk = np.sum(original_vec_cpu[data_indices] != binarized_extracted[data_indices])
                    total_bit_errors += bit_errors_in_chunk
                    total_bits_tested += len(data_indices)

                # --- Calculate SNR ---
                with torch.no_grad():
                    # Generate a "clean" noise audio for comparison
                    noise_vec = torch.randn(1, LATENT_DIM, device=DEVICE)
                    clean_audio = generator(noise_vec)
                    signal_power = torch.mean(clean_audio.pow(2))
                    noise_power = torch.mean((generated_audio - clean_audio).pow(2))
                    snr = 10 * torch.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                    total_snr += snr.item()
                    total_chunks_processed += 1

                # Convert back to data
                extracted_fec_chunk = binary_vector_to_data(extracted_vector.cpu())

                try:
                    # The library expects a mutable bytearray. This is a critical detail.
                    data_byte_array = bytearray(extracted_fec_chunk)
                    corrected_chunk, _, _ = rs.decode(data_byte_array)
                    reconstructed_payload += corrected_chunk
                except Exception: # Catches Reed-Solomon errors if chunk is too corrupted
                    is_corrupted = True
                    break # This chunk is unrecoverable, so the message is lost.

            except Exception: # This catches errors in vector conversion
                is_corrupted = True
                break # No need to process further chunks if one fails

        # 3. Check if the final message is correct
        if is_corrupted or reconstructed_payload != original_message_bytes:
            total_message_errors += 1

    print("✅ Evaluation complete.")

    # --- Final Results ---
    mer = (total_message_errors / NUM_TEST_MESSAGES) if NUM_TEST_MESSAGES > 0 else 0
    ber = (total_bit_errors / total_bits_tested) if total_bits_tested > 0 else 0
    avg_snr = (total_snr / total_chunks_processed) if total_chunks_processed > 0 else 0

    print("\n--- Final Metrics (with FEC) ---")
    print(f"Total Messages Tested: {NUM_TEST_MESSAGES}")
    print("-----------------------")
    print(f"📈 Average Signal-to-Noise Ratio (SNR): {avg_snr:.2f} dB")
    print(f"📉 Bit Error Rate (BER) [pre-FEC]:    {ber:.6f} ({total_bit_errors}/{total_bits_tested} bit errors)")
    print(f"📊 Final Message Error Rate (MER):  {mer:.6f} ({total_message_errors} errors)")
    print("-----------------------")
    print("SNR: Higher is better (less audible distortion).")
    print("BER: The raw error rate of the neural network before correction.")
    print("MER: The final, end-to-end reliability of the system after error correction.")

    # Return the metrics as a dictionary so other scripts can use them
    return {
        "Total Messages Tested": NUM_TEST_MESSAGES,
        "Average SNR (dB)": f"{avg_snr:.2f}",
        "Bit Error Rate (BER)": f"{ber:.6f}",
        "Message Errors": total_message_errors,
        "Final Message Error Rate (MER)": f"{mer:.6f}",
    }


if __name__ == "__main__":
    main()