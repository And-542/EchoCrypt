import torch
import argparse
import sys
import os
import scipy.io.wavfile
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reedsolo import RSCodec
from models.generator import Generator
from utils.conversion import data_to_binary_vector

# --- Configuration (must match training and app config) ---
LATENT_DIM = 256
FEC_SYMBOLS = 24
MAX_CHUNK_SIZE = (LATENT_DIM - 8) // 8
MAX_PAYLOAD_BYTES_PER_CHUNK = MAX_CHUNK_SIZE - FEC_SYMBOLS

GEN_MODEL_PATH = "d:/EchoCrypt/EchoCrypt/models/saved_models/generator_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000

def hide_message(message: str, output_path: str):
    """
    Encodes a message into a WAV audio file using the Generator model.

    Args:
        message (str): The message to hide.
        output_path (str): The path to save the output WAV file.
    """
    print("--- EchoCrypt: Hiding Message ---")
    # 1. Load Model and FEC
    print(f"💿 Loading Generator model on {DEVICE}...")
    generator = Generator(latent_dim=LATENT_DIM)
    generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    generator.to(DEVICE)
    generator.eval()
    rs = RSCodec(FEC_SYMBOLS)
    print("✅ Model loaded.")

    # 2. Prepare and Chunk the Message
    message_bytes = message.encode('utf-8')
    payload_chunks = [message_bytes[i:i + MAX_PAYLOAD_BYTES_PER_CHUNK] for i in range(0, len(message_bytes), MAX_PAYLOAD_BYTES_PER_CHUNK)]

    print(f"✉️  Message: '{message}' ({len(message_bytes)} bytes)")
    print(f"📦 Splitting into {len(payload_chunks)} chunk(s).")

    final_audio = np.array([], dtype=np.float32)

    # 3. Process each chunk
    for i, chunk in enumerate(payload_chunks):
        print(f"   - Processing chunk {i+1}/{len(payload_chunks)}...")
        try:
            # Add FEC
            fec_chunk = rs.encode(chunk)

            # Convert to vector
            data_vector = data_to_binary_vector(fec_chunk, LATENT_DIM).to(DEVICE)

            # Generate audio
            with torch.no_grad():
                generated_audio_tensor = generator(data_vector)

            # Convert to numpy array and append
            audio_chunk_np = generated_audio_tensor.squeeze().cpu().numpy()
            final_audio = np.concatenate((final_audio, audio_chunk_np))

        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}")
            print("Aborting.")
            return

    # 4. Save the final audio file
    # Normalize to 16-bit integer range
    final_audio_int16 = np.int16(final_audio / np.max(np.abs(final_audio)) * 32767)
    scipy.io.wavfile.write(output_path, SAMPLE_RATE, final_audio_int16)
    print(f"\n✅ Success! Audio saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hide a secret message in a WAV audio file.")
    parser.add_argument("message", type=str, help="The secret message to hide.")
    parser.add_argument(
        "--output",
        type=str,
        default="stego_audio.wav",
        help="Path to save the output WAV file."
    )
    args = parser.parse_args()

    hide_message(args.message, args.output)