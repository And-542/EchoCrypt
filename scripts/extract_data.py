import sys
import os
import torch
import soundfile as sf
import argparse
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.extractor import Extractor

def binary_vector_to_message(vector: torch.Tensor) -> str:
    """
    Converts a binary latent vector back into a string message.

    Args:
        vector (torch.Tensor): The latent vector recovered by the extractor.

    Returns:
        str: The decoded message.
    """
    # 1. Apply a threshold to convert float values to bits (0 or 1)
    # If value > 0, it's a '1', otherwise it's a '0'
    binary_string = ''.join(['1' if val > 0 else '0' for val in vector.squeeze()])

    # 2. Extract the length prefix (first 8 bits)
    length_prefix_bits = 8
    if len(binary_string) < length_prefix_bits:
        return "[Error: Incomplete vector]"
    
    length_binary = binary_string[:length_prefix_bits]
    message_length = int(length_binary, 2)

    # 3. Extract the message itself
    message_binary = binary_string[length_prefix_bits : length_prefix_bits + message_length * 8]
    byte_chunks = [message_binary[i:i+8] for i in range(0, len(message_binary), 8)]

    # 4. Convert each byte back to a character and build the message
    message = ""
    for byte in byte_chunks:
        if len(byte) == 8:
            char_code = int(byte, 2)
            message += chr(char_code)
    return message

def extract_message_from_audio(audio_file_path: str, model: Extractor, device: str, sample_rate: int, audio_length_samples: int) -> str:
    """
    Extracts a secret message from an audio file using the Extractor model.

    Args:
        audio_file_path (str): Path to the audio file to check.
        message (str): The secret message to test for.
        model (Extractor): The trained extractor model.
        device (str): The device to run the model on ('cuda' or 'cpu').
        sample_rate (int): The sample rate of the audio.
        audio_length_samples (int): The number of samples per message chunk.

    Returns:
        str: The extracted secret message.
    """
    try:
        received_audio, sr = sf.read(audio_file_path, dtype='float32')
        if sr != sample_rate:
            print(f"⚠️ Warning: Audio file sample rate ({sr}) differs from model sample rate ({sample_rate}).")
            # In a real-world scenario, you might want to resample here.
            # For now, we'll proceed but the comparison will likely fail.
        print(f"🔍 Loaded audio file: {audio_file_path}")
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return ""

    # --- Process Audio in Chunks ---
    num_chunks = len(received_audio) // audio_length_samples
    if len(received_audio) % audio_length_samples != 0:
        print(f"⚠️ Warning: Audio file length is not a perfect multiple of the chunk size ({audio_length_samples} samples). The last partial chunk will be ignored.")

    print(f"Audio contains {num_chunks} potential message chunk(s).")
    full_message = ""

    for i in range(num_chunks):
        print(f"--- Processing Chunk {i+1}/{num_chunks} ---")
        chunk_audio = received_audio[i * audio_length_samples : (i + 1) * audio_length_samples]
        
        received_audio_tensor = torch.from_numpy(chunk_audio).to(device)
        received_audio_tensor = received_audio_tensor.unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            extracted_vector = model(received_audio_tensor)

        decoded_chunk = binary_vector_to_message(extracted_vector)
        print(f"Decoded chunk: '{decoded_chunk}'")
        full_message += decoded_chunk

    return full_message

def main():
    parser = argparse.ArgumentParser(description="Extract a secret message from GAN-generated audio.")
    parser.add_argument("audio_file", type=str, help="Path to the .wav file to check.")
    # The message argument is no longer needed for extraction
    args = parser.parse_args()

    # --- Configuration ---
    LATENT_DIM = 256
    MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\extractor_final.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_RATE = 16000
    AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1

    # --- Load Model ---
    print(f"💿 Loading trained extractor model from {MODEL_PATH}")
    model = Extractor(input_length=AUDIO_LENGTH_SAMPLES, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # --- Extract Message ---
    extracted_message = extract_message_from_audio(args.audio_file, model, DEVICE, SAMPLE_RATE, AUDIO_LENGTH_SAMPLES)

    print("\n--- Result ---")
    print(f"💬 Extracted Message: '{extracted_message}'")

if __name__ == "__main__":
    main()