import torch
import argparse
import sys
import os
import scipy.io.wavfile
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reedsolo import RSCodec
from models.extractor import Extractor
from utils.conversion import binary_vector_to_data

# --- Configuration (must match training and app config) ---
LATENT_DIM = 256
FEC_SYMBOLS = 16

EXT_MODEL_PATH = "d:/EchoCrypt/EchoCrypt/models/saved_models/extractor_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
AUDIO_CHUNK_SAMPLES = SAMPLE_RATE * 1 # Each chunk is 1 second

def extract_message(input_path: str):
    """
    Extracts a hidden message from a WAV audio file using the Extractor model.

    Args:
        input_path (str): The path to the input WAV file.
    """
    print("--- EchoCrypt: Extracting Message ---")
    # 1. Load Model and FEC
    print(f"💿 Loading Extractor model on {DEVICE}...")
    extractor = Extractor(latent_dim=LATENT_DIM)
    extractor.load_state_dict(torch.load(EXT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    extractor.to(DEVICE)
    extractor.eval()
    rs = RSCodec(FEC_SYMBOLS)
    print("✅ Model loaded.")

    # 2. Load and Chunk Audio
    try:
        rate, audio_data = scipy.io.wavfile.read(input_path)
        if rate != SAMPLE_RATE:
            print(f"⚠️ Warning: Audio sample rate is {rate}Hz, but model expects {SAMPLE_RATE}Hz. Results may be poor.")

        # Normalize to float32 range [-1, 1]
        audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        audio_chunks = [audio_float[i:i + AUDIO_CHUNK_SAMPLES] for i in range(0, len(audio_float), AUDIO_CHUNK_SAMPLES)]
        print(f"🎵 Loaded '{input_path}'. Found {len(audio_chunks)} potential data chunk(s).")
    except Exception as e:
        print(f"❌ Failed to load audio file: {e}")
        return

    # 3. Process each chunk
    reconstructed_payload = b""
    for i, chunk in enumerate(audio_chunks):
        if len(chunk) < AUDIO_CHUNK_SAMPLES:
            print(f"   - Skipping final chunk {i+1} as it's too short.")
            continue

        print(f"   - Processing chunk {i+1}/{len(audio_chunks)}...")
        audio_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(1).to(DEVICE) # Shape: (1, 1, 16000)

        with torch.no_grad():
            extracted_vector = extractor(audio_tensor)

        extracted_fec_chunk = binary_vector_to_data(extracted_vector.cpu())

        try:
            data_byte_array = bytearray(extracted_fec_chunk)
            corrected_chunk, _, _ = rs.decode(data_byte_array)
            reconstructed_payload += corrected_chunk
        except Exception:
            print(f"   - ⚠️ Chunk {i+1} was too corrupted to recover. Message may be incomplete.")

    # 4. Display result
    print("\n--- Extraction Complete ---")
    print(f"✉️ Reconstructed Message: {reconstructed_payload.decode('utf-8', errors='ignore')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a secret message from a WAV audio file.")
    parser.add_argument("input_file", type=str, help="Path to the input WAV file.")
    args = parser.parse_args()
    extract_message(args.input_file)