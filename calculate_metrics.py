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

def data_to_binary_vector_silent(data_bytes: bytes, latent_dim: int) -> torch.Tensor:
    """
    Converts a byte string into a binary latent vector without printing logs.
    """
    HEADER_BITS = 8
    max_payload_bytes = (latent_dim - HEADER_BITS) // 8
    if len(data_bytes) > max_payload_bytes:
        raise ValueError(f"Data payload is too long for one chunk. Max length: {max_payload_bytes} bytes.")

    binary_payload = ''.join(format(byte, '08b') for byte in data_bytes)
    length_binary = format(len(data_bytes), f'0{HEADER_BITS}b')
    full_binary_string = length_binary + binary_payload
    binary_values = [1.0 if bit == '1' else -1.0 for bit in full_binary_string]
    padding_size = latent_dim - len(binary_values)
    padded_vector = binary_values + [0.0] * padding_size
    return torch.tensor(padded_vector, dtype=torch.float32).unsqueeze(0)

def binary_vector_to_data_silent(vector: torch.Tensor) -> bytes:
    """Converts a binary latent vector back into a byte string."""
    binary_string = ''.join(['1' if val > 0 else '0' for val in vector.squeeze()])
    if len(binary_string) < 8: return b""
    length_binary = binary_string[:8]
    try:
        message_length = int(length_binary, 2)
    except ValueError:
        return b"" # Invalid length prefix
        
    # The total number of bits to read is the header (8) + the payload bits.
    total_bits_to_read = 8 + message_length * 8
    if len(binary_string) < total_bits_to_read:
        return b"" # Not enough data to form a full message

    # Slice the exact portion of the binary string that represents the data.
    data_binary = binary_string[8:total_bits_to_read]
    byte_chunks = [data_binary[i:i+8] for i in range(0, len(data_binary), 8)]

    try:
        # Use a robust method to convert binary strings to bytes
        return b"".join(int(b, 2).to_bytes(1, 'big') for b in byte_chunks)
    except (ValueError, OverflowError):
        return b""

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
                original_vector = data_to_binary_vector_silent(fec_chunk, LATENT_DIM).to(DEVICE)

                # Autoencoder pass
                with torch.no_grad():
                    generated_audio = generator(original_vector)
                    extracted_vector = extractor(generated_audio) # Shape is already (1, 1, 16000)

                # Convert back to data
                extracted_fec_chunk = binary_vector_to_data_silent(extracted_vector.cpu())

                # Perform error correction
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

    print("\n--- Final Metrics (with FEC) ---")
    print(f"Total Messages Tested: {NUM_TEST_MESSAGES}")
    print("-----------------------")
    print(f"📊 Final Message Error Rate (MER):  {mer:.6f} ({total_message_errors} errors)")
    print("-----------------------")
    print("This MER reflects the end-to-end reliability of the system, including error correction.")

    # Return the metrics as a dictionary so other scripts can use them
    return {
        "Total Messages Tested": NUM_TEST_MESSAGES,
        "Message Errors": total_message_errors,
        "Final Message Error Rate (MER)": f"{mer:.6f}"
    }


if __name__ == "__main__":
    main()