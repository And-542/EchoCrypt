import gradio as gr
import torch
import numpy as np
import librosa # Use librosa for more robust audio loading
import os
import sys
from scipy.io.wavfile import write as write_wav
import tempfile

# Encryption Modules---
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import scrypt
from reedsolo import RSCodec # For Forward Error Correction

# Add the project root to the Python path to allow importing project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generator import Generator
from models.extractor import Extractor
from utils.conversion import data_to_binary_vector, binary_vector_to_data

# --- Configuration ---
LATENT_DIM = 256 # Must match the trained models
GEN_MODEL_PATH = "d:/EchoCrypt/EchoCrypt/models/saved_models/generator_final.pth"
EXT_MODEL_PATH = "d:/EchoCrypt/EchoCrypt/models/saved_models/extractor_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1
FEC_SYMBOLS = 24 # Number of error correction bytes to add per chunk

# --- Load Models (do this once on startup) ---
print(f"Running on device: {DEVICE}")

print(f"💿 Loading Generator model from {GEN_MODEL_PATH}")
generator = Generator(latent_dim=LATENT_DIM)
generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE, weights_only=True))
generator.to(DEVICE)
generator.eval()

print(f"💿 Loading Extractor model from {EXT_MODEL_PATH}")
extractor = Extractor(latent_dim=LATENT_DIM)
extractor.load_state_dict(torch.load(EXT_MODEL_PATH, map_location=DEVICE, weights_only=True))
extractor.to(DEVICE)
extractor.eval()

print("✅ Models loaded successfully.")

# --- Warmup Models ---
# This prevents the first user request from being slow due to CUDA initialization
print("🔥 Warming up models and audio processing...")
with torch.no_grad():
    dummy_latent = torch.randn(1, LATENT_DIM).to(DEVICE)
    dummy_audio = generator(dummy_latent)
    extractor(dummy_audio)
    
    # Warmup Librosa (triggers Numba JIT compilation and backend initialization)
    dummy_audio_np = dummy_audio.squeeze().cpu().numpy()
    # Create a temp file to force librosa to go through file loading logic
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write_wav(tmp.name, SAMPLE_RATE, dummy_audio_np)
        tmp_path = tmp.name
    
    try:
        librosa.load(tmp_path, sr=SAMPLE_RATE)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
print("✅ Warmup complete.")

# --- Helper Functions (adapted from scripts) ---

def get_key_from_password(password: str, salt: bytes) -> bytes:
    """Derives a 32-byte AES key from a password using scrypt."""
    return scrypt(password, salt, key_len=32, N=2**14, r=8, p=1)

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
    # Each chunk will have data + FEC symbols.
    # The total size of a chunk must fit into our latent vector.
    max_chunk_size = (LATENT_DIM - 8) // 8 # Max bytes per vector (e.g., 11)
    max_payload_bytes = max_chunk_size - FEC_SYMBOLS # Bytes available for actual data
    rs = RSCodec(FEC_SYMBOLS)
    payload_chunks = [payload[i:i + max_payload_bytes] for i in range(0, len(payload), max_payload_bytes)]
    
    all_audio_chunks = []
    
    for chunk in payload_chunks:
        try:
            # Convert the data chunk (bytes) to a latent vector
            # Add FEC to the chunk before converting to a vector
            fec_chunk = rs.encode(chunk)
            latent_vector = data_to_binary_vector(fec_chunk, LATENT_DIM).to(DEVICE)

        except ValueError as e:
            raise gr.Error(str(e))

        with torch.no_grad():
            generated_waveform = generator(latent_vector)
        
        audio_data = generated_waveform.squeeze().cpu().numpy()
        all_audio_chunks.append(audio_data)

    if not all_audio_chunks:
        # This case should not be hit with correct logic, but it's a safe guard.
        raise gr.Error("Failed to generate any audio chunks. The message might be empty or too short.")

    final_audio = np.concatenate(all_audio_chunks)
    
    # Gradio handles temporary file creation for outputs
    output_path = os.path.abspath("embedded_message.wav")
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

    # Use librosa to read the audio file. It can handle many more formats, including mp4.
    # We specify the target sample rate to automatically resample.
    try:
        received_audio, sr = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return f"[Error] Failed to load or process audio file. It might be an unsupported format or corrupted. Details: {e}"

    if sr != SAMPLE_RATE:
        # This check is redundant if librosa resampling works, but good for safety.
        return f"[Error] Audio sample rate ({sr}Hz) does not match model's required rate ({SAMPLE_RATE}Hz)."

    if received_audio.dtype != np.float32:
        # Normalize to float32 if it's an integer type
        received_audio = received_audio.astype(np.float32) / np.iinfo(received_audio.dtype).max

    num_chunks = len(received_audio) // AUDIO_LENGTH_SAMPLES
    if num_chunks == 0:
        return "[Error] Audio file is too short to contain a message."

    # --- Optimization: Batch Processing ---
    # Instead of looping and running the model N times, we run it once with batch size N.
    
    # 1. Truncate audio to exact multiple of chunk size
    valid_length = num_chunks * AUDIO_LENGTH_SAMPLES
    received_audio = received_audio[:valid_length]

    # 2. Reshape to (Batch_Size, 1, Samples)
    # Use torch.tensor to copy data, ensuring contiguity and avoiding stride issues with numpy slices
    input_tensor = torch.tensor(received_audio, dtype=torch.float32, device=DEVICE).view(num_chunks, 1, AUDIO_LENGTH_SAMPLES)

    # 3. Run inference on the whole batch
    with torch.no_grad():
        extracted_vectors = extractor(input_tensor) # Output shape: (num_chunks, latent_dim)
    
    # 4. Move results to CPU for decoding
    extracted_vectors_cpu = extracted_vectors.cpu()

    full_payload = b""
    for i in range(num_chunks):
        # Process the vector from the batch
        fec_chunk = binary_vector_to_data(extracted_vectors_cpu[i])
        
        # --- Error Correction ---
        try:
            rs = RSCodec(FEC_SYMBOLS)
            # The library expects a mutable bytearray.
            data_byte_array = bytearray(fec_chunk)
            corrected_chunk, _, _ = rs.decode(data_byte_array)
            full_payload += corrected_chunk

        except Exception: # Catches Reed-Solomon errors if chunk is too corrupted
            return "[Extraction Failed] Data is too corrupted to be recovered, even with FEC."

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

# Custom Theme and CSS for a modern look

custom_css = """
body, .gradio-container {
    background-color: #05050a !important;
    color: #e9d5ff !important;
}
h1 {
    text-align: center;
    font-weight: 900 !important;
    background: -webkit-linear-gradient(45deg, #a855f7, #d946ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
    filter: drop-shadow(0 0 10px rgba(168, 85, 247, 0.5));
}
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #d8b4fe;
    margin-bottom: 2rem;
    font-weight: 500;
    text-shadow: 0 0 5px rgba(216, 180, 254, 0.4);
}
.tabs {
    margin-top: 1rem;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.2);
    border: 1px solid #581c87;
    background-color: #110a1f !important;
}
footer {
    text-align: center;
    margin-top: 3rem;
    color: #a855f7;
    font-size: 0.8rem;
    opacity: 0.8;
}
/* Neon overrides */
button.primary {
    background: linear-gradient(90deg, #7e22ce, #c026d3) !important;
    box-shadow: 0 0 15px rgba(192, 38, 211, 0.5) !important;
    border: none !important;
    color: white !important;
}
label, span {
    color: #e9d5ff !important;
}
/* Input styling */
input, textarea {
    background-color: #130e26 !important;
    border: 1px solid #6b21a8 !important;
    color: #e9d5ff !important;
}
input:focus, textarea:focus {
    border-color: #d946ef !important;
    box-shadow: 0 0 10px rgba(217, 70, 239, 0.3) !important;
}
"""

with gr.Blocks() as demo:
    gr.HTML("<style>" + custom_css + "</style>")
    with gr.Column():
        gr.Markdown(
            """
            # 🎧 EchoCrypt
            <div class="subtitle">Coverless White Noise Audio Steganography</div>
            """
        )

        with gr.Tabs(elem_classes="tabs"):
            with gr.Tab("🔒 Embed Message"):
                with gr.Row():
                    with gr.Column(scale=1):
                        embed_input = gr.Textbox(
                            label="Secret Message", 
                            placeholder="Type your confidential message here...", 
                            lines=4
                        )
                        with gr.Row():
                            embed_password = gr.Textbox(
                                label="Encryption Password", 
                                placeholder="Enter a strong password", 
                                type="password", 
                                scale=10
                            )
                            toggle_embed_vis = gr.Button("👁️", scale=1, min_width=10)
                        embed_password_visible = gr.State(False)

                        embed_button = gr.Button("Generate Secure Audio 🎵", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Output")
                        embed_output_audio = gr.Audio(label="Stego-Audio", type="filepath", interactive=False)
                        download_file = gr.File(label="Download WAV", visible=False, interactive=False)

                embed_button.click(
                    fn=embed_message,
                    inputs=[embed_input, embed_password],
                    outputs=[embed_output_audio, download_file]
                )
                toggle_embed_vis.click(
                    fn=toggle_password_visibility,
                    inputs=[embed_password_visible],
                    outputs=[embed_password_visible, embed_password, toggle_embed_vis]
                )

            with gr.Tab("🔓 Extract Message"):
                with gr.Row():
                    with gr.Column(scale=1):
                        extract_input_file = gr.File(
                            label="Upload Audio File",
                            file_types=[".wav", ".mp3", ".mp4", ".flac"],
                            type="filepath",
                            height=200
                        )
                        with gr.Row():
                            extract_password = gr.Textbox(
                                label="Decryption Password", 
                                placeholder="Enter the password used for encryption", 
                                type="password", 
                                scale=10
                            )
                            toggle_extract_vis = gr.Button("👁️", scale=1, min_width=10)
                        extract_password_visible = gr.State(False)

                        extract_button = gr.Button("Decrypt & Extract 🔓", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📩 Decoded Message")
                        extract_output_text = gr.Textbox(
                            label="Result", 
                            placeholder="The extracted message will appear here...", 
                            lines=6
                        )

                extract_button.click(
                    fn=extract_message,
                    inputs=[extract_input_file, extract_password],
                    outputs=extract_output_text
                )
                toggle_extract_vis.click(
                    fn=toggle_password_visibility,
                    inputs=[extract_password_visible],
                    outputs=[extract_password_visible, extract_password, toggle_extract_vis]
                )

        gr.Markdown(
            """
            <footer>
            EchoCrypt Project • Powered by PyTorch, Gradio & AES-256 Encryption
            </footer>
            """
        )

if __name__ == "__main__":
    # To make the UI accessible on your local network, set share=True
    demo.launch()