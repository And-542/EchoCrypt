import sys
import os
import torch
import soundfile as sf
import argparse
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import Generator


def message_to_binary_vector(message: str, latent_dim: int) -> torch.Tensor:
    """
    Converts a string message into a binary latent vector for true extraction.

    Args:
        message (str): The secret message to embed.
        latent_dim (int): The dimensionality of the latent vector.

    Returns:
        torch.Tensor: A tensor of shape (1, latent_dim) with values -1 or 1.
    """
    # 1. Define the space for the length prefix (1 byte = 8 bits)
    length_prefix_bits = 8
    max_message_chars = (latent_dim - length_prefix_bits) // 8

    # 2. Check if the message is too long
    if len(message) > max_message_chars:
        raise ValueError(f"Message is too long. Max length: {max_message_chars} characters. Message is {len(message)} characters.")

    # 3. Create the length prefix and message binary strings
    length_binary = format(len(message), f'0{length_prefix_bits}b')
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    
    # 4. Combine length prefix and message
    full_binary_string = length_binary + binary_message

    # 5. Map binary string to a tensor of -1s and 1s
    # We use -1 for '0' and 1 for '1' as it's a common practice in ML
    binary_values = [1.0 if bit == '1' else -1.0 for bit in full_binary_string]

    # 6. Pad the rest of the vector with a neutral value (e.g., 0)
    padding_size = latent_dim - len(binary_values)
    padded_vector = binary_values + [0.0] * padding_size

    print(f"🌱 Message '{message}' ({len(message)} chars) converted to a binary vector.")
    print(f"   (Using {length_prefix_bits} bits for length, {len(binary_message)} bits for data)")
    return torch.tensor(padded_vector, dtype=torch.float32).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Embed a secret message into GAN-generated audio.")
    # Default output path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output = os.path.join(project_root, 'data', 'noise', 'embedded_message.wav')
    parser.add_argument("-o", "--output", type=str, default=default_output, help="Path to save the output .wav file.")
    args = parser.parse_args()

    # --- Get Message from User ---
    try:
        message_to_embed = input("Please enter the secret message to embed: ")
        if not message_to_embed:
            print("❌ Error: Message cannot be empty.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Embedding cancelled by user.")
        return

    # --- Configuration ---
    LATENT_DIM = 256
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

    # --- Chunk Message and Generate Audio ---
    length_prefix_bits = 8
    max_chars_per_chunk = (LATENT_DIM - length_prefix_bits) // 8
    message_chunks = [message_to_embed[i:i + max_chars_per_chunk] for i in range(0, len(message_to_embed), max_chars_per_chunk)]

    print(f"\nMessage is {len(message_to_embed)} characters long, splitting into {len(message_chunks)} chunk(s).")

    all_audio_chunks = []

    for i, chunk in enumerate(message_chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(message_chunks)} ---")
        
        # --- Generate Latent Vector from Message Chunk ---
        try:
            latent_vector = message_to_binary_vector(chunk, LATENT_DIM).to(DEVICE)
        except ValueError as e:
            print(f"❌ Error processing chunk: {e}")
            continue

        # --- Generate Audio ---
        print("🎶 Generating audio from message chunk...")
        with torch.no_grad():
            generated_waveform = model(latent_vector)

        # --- Collect Audio Chunk ---
        audio_data = generated_waveform.squeeze().cpu().numpy()
        all_audio_chunks.append(audio_data)
        print(f"✅ Audio for chunk {i+1} generated.")

    # --- Concatenate and Save Final Audio ---
    final_audio = np.concatenate(all_audio_chunks)
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, final_audio, samplerate=SAMPLE_RATE)
    print(f"\n✅ Full audio with embedded message saved to {output_path}")

if __name__ == "__main__":
    main()
