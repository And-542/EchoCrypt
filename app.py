import gradio as gr
import torch
import numpy as np
import soundfile as sf
import os
import sys
from scipy.io.wavfile import write as write_wav

# --- Novelty: Add Encryption ---
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import scrypt

# Add the project root to the Python path to allow importing project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generator import Generator
from models.extractor import Extractor

# --- Configuration ---
LATENT_DIM = 100
GEN_MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\generator_final.pth"
EXT_MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\extractor_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1

# --- Load Models (do this once on startup) ---
print(f"Running on device: {DEVICE}")

print(f"💿 Loading Generator model from {GEN_MODEL_PATH}")
generator = Generator(latent_dim=LATENT_DIM, output_length=AUDIO_LENGTH_SAMPLES)
generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE, weights_only=True))
generator.to(DEVICE)
generator.eval()

print(f"💿 Loading Extractor model from {EXT_MODEL_PATH}")
extractor = Extractor(input_length=AUDIO_LENGTH_SAMPLES, latent_dim=LATENT_DIM)
extractor.load_state_dict(torch.load(EXT_MODEL_PATH, map_location=DEVICE, weights_only=True))
extractor.to(DEVICE)
extractor.eval()

print("✅ Models loaded successfully.")

# --- Helper Functions (adapted from scripts) ---

def get_key_from_password(password: str, salt: bytes) -> bytes:
    """Derives a 32-byte AES key from a password using scrypt."""
    return scrypt(password, salt, key_len=32, N=2**14, r=8, p=1)

def data_to_binary_vector(data_bytes: bytes, latent_dim: int) -> torch.Tensor:
    """Converts a byte string (e.g., ciphertext) into a binary latent vector."""
    # 8 bits for length prefix
    HEADER_BITS = 8
    max_payload_bytes = (latent_dim - HEADER_BITS) // 8
    if len(data_bytes) > max_payload_bytes:
        raise ValueError(f"Data payload is too long for one chunk. Max length: {max_payload_bytes} bytes.")

    # Convert payload to binary string
    binary_payload = ''.join(format(byte, '08b') for byte in data_bytes)
    
    # Create length prefix for the payload
    length_binary = format(len(data_bytes), f'08b')

    # The full binary string is length + payload
    full_binary_string = length_binary + binary_payload
    binary_values = [1.0 if bit == '1' else -1.0 for bit in full_binary_string]
    
    # Pad the rest of the vector
    padding_size = latent_dim - len(binary_values)
    padded_vector = binary_values + [0.0] * padding_size
    return torch.tensor(padded_vector, dtype=torch.float32).unsqueeze(0)

def binary_vector_to_data(vector: torch.Tensor) -> bytes:
    """Converts a binary latent vector back into a byte string."""
    binary_string = ''.join(['1' if val > 0 else '0' for val in vector.squeeze()])
    
    # Extract length prefix (8 bits)
    if len(binary_string) < 8:
        return b""
    length_binary = binary_string[:8]
    message_length = int(length_binary, 2)

    # Extract the payload
    message_binary = binary_string[8 : 8 + message_length * 8]
    byte_chunks = [message_binary[i:i+8] for i in range(0, len(message_binary), 8)]
    
    data_bytes = bytearray()
    for byte in byte_chunks:
        if len(byte) == 8:
            data_bytes.append(int(byte, 2))
    return bytes(data_bytes)

# --- Gradio Interface Functions ---

def embed_message(message: str, password: str):
    """Gradio function to embed a message and return the audio file path for multiple components."""
    if not message:
        raise gr.Error("Message cannot be empty.")
    if not password:
        raise gr.Error("Password cannot be empty.")

    # --- Encryption ---
    salt = os.urandom(16) # Generate a new random salt for each message
    key = get_key_from_password(password, salt)
    cipher = AES.new(key, AES.MODE_CBC) # IV is generated automatically
    
    # Encrypt the message (must be bytes)
    message_bytes = message.encode('utf-8')
    ciphertext = cipher.encrypt(pad(message_bytes, AES.block_size))
    
    # The payload is salt + iv + ciphertext. This is what we hide.
    payload = salt + cipher.iv + ciphertext

    # --- Chunking the Payload ---
    # 8 bits for length prefix. The rest is for the payload.
    max_payload_bytes = (LATENT_DIM - 8) // 8
    payload_chunks = [payload[i:i + max_payload_bytes] for i in range(0, len(payload), max_payload_bytes)]
    
    all_audio_chunks = []
    
    for chunk in payload_chunks:
        try:
            # Convert the data chunk (bytes) to a latent vector
            latent_vector = data_to_binary_vector(chunk, LATENT_DIM).to(DEVICE)
        except ValueError as e:
            raise gr.Error(str(e))

        with torch.no_grad():
            generated_waveform = generator(latent_vector)
        
        audio_data = generated_waveform.squeeze().cpu().numpy()
        all_audio_chunks.append(audio_data)

    final_audio = np.concatenate(all_audio_chunks)
    
    # Gradio handles temporary file creation for outputs
    output_path = "embedded_message.wav"
    write_wav(output_path, SAMPLE_RATE, final_audio.astype(np.float32))
    
    # We need to return a value for each output component defined in the .click() event.
    # The first path goes to the gr.Audio component.
    # The second path goes to the gr.File component, which we also make visible.
    return output_path, gr.update(value=output_path, visible=True)

def extract_message(audio_filepath, password: str):
    """Gradio function to extract a message from an uploaded audio file."""
    if audio_filepath is None:
        raise gr.Error("Please upload an audio file.")
    if not password:
        raise gr.Error("Password cannot be empty.")

    # Read the audio file using soundfile since we get a filepath
    received_audio, sr = sf.read(audio_filepath, dtype='float32')

    if sr != SAMPLE_RATE:
        # In a real app, you'd resample. For now, we'll raise an error.
        return f"[Error] Audio sample rate ({sr}Hz) does not match model's required rate ({SAMPLE_RATE}Hz)."

    if received_audio.dtype != np.float32:
        # Normalize to float32 if it's an integer type
        received_audio = received_audio.astype(np.float32) / np.iinfo(received_audio.dtype).max

    num_chunks = len(received_audio) // AUDIO_LENGTH_SAMPLES
    if num_chunks == 0:
        return "[Error] Audio file is too short to contain a message."

    full_payload = b""
    for i in range(num_chunks):
        chunk_audio = received_audio[i * AUDIO_LENGTH_SAMPLES : (i + 1) * AUDIO_LENGTH_SAMPLES]
        received_audio_tensor = torch.from_numpy(chunk_audio).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            extracted_vector = extractor(received_audio_tensor)
        
        # Convert vector back to data bytes
        payload_chunk = binary_vector_to_data(extracted_vector)
        full_payload += payload_chunk

    if not full_payload:
        return "[No data found in audio]"

    # --- Decryption ---
    try:
        salt = full_payload[:16]
        iv = full_payload[16:32]
        ciphertext = full_payload[32:]

        key = get_key_from_password(password, salt)
        cipher = AES.new(key, AES.MODE_CBC, iv=iv)
        decrypted_message_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted_message_bytes.decode('utf-8')
    except (ValueError, KeyError):
        return "[Decryption Failed] Incorrect password or corrupted data."

def toggle_password_visibility(is_visible):
    """Toggles the visibility of a password field."""
    if is_visible:
        # If currently visible, hide it
        new_type = "password"
        new_icon = "👁️" # Eye icon for "show"
    else:
        # If currently hidden, show it
        new_type = "text"
        new_icon = "🙈" # Monkey icon for "hide"
    
    return not is_visible, gr.update(type=new_type), gr.update(value=new_icon)

# --- Build and Launch the Gradio App ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎧 EchoCrypt: Audio Steganography 
        Use the tools below to hide a message in a generated audio file or extract a message from an existing one.
        """
    )

    with gr.Tab("Embed Message"):
        with gr.Row():
            with gr.Column():
                embed_input = gr.Textbox(label="Secret Message", placeholder="Enter your secret message here...")
                with gr.Row():
                    embed_password = gr.Textbox(label="Password", placeholder="Enter a password for encryption", type="password", container=False, scale=10)
                    toggle_embed_vis = gr.Button("👁️", min_width=10, scale=1)
                embed_password_visible = gr.State(False)

                embed_button = gr.Button("Generate Audio", variant="primary")
            with gr.Column():
                embed_output_audio = gr.Audio(label="Generated Audio with Hidden Message", type="filepath", show_download_button=True)
                download_file = gr.File(label="Download Audio File", visible=False)
        embed_button.click(
            fn=embed_message,
            inputs=[embed_input, embed_password],
            # The function returns the same path to both the audio player and the file download component
            outputs=[embed_output_audio, download_file]
        )
        toggle_embed_vis.click(
            fn=toggle_password_visibility,
            inputs=[embed_password_visible],
            outputs=[embed_password_visible, embed_password, toggle_embed_vis]
        )

    with gr.Tab("Extract Message"):
        with gr.Row():
            with gr.Column():
                extract_input_audio = gr.Audio(label="Upload Audio File", type="filepath")
                with gr.Row():
                    extract_password = gr.Textbox(label="Password", placeholder="Enter the password used for encryption", type="password", container=False, scale=10)
                    toggle_extract_vis = gr.Button("👁️", min_width=10, scale=1)
                extract_password_visible = gr.State(False)

                extract_button = gr.Button("Extract Message", variant="primary")
            with gr.Column():
                extract_output_text = gr.Textbox(label="Extracted Message")
        extract_button.click(
            fn=extract_message,
            inputs=[extract_input_audio, extract_password],
            outputs=extract_output_text
        )
        toggle_extract_vis.click(
            fn=toggle_password_visibility,
            inputs=[extract_password_visible],
            outputs=[extract_password_visible, extract_password, toggle_extract_vis]
        )

if __name__ == "__main__":
    # To make the UI accessible on your local network, set share=True
    demo.launch()