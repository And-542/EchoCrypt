import gradio as gr
import torch
import numpy as np
import soundfile as sf
import os
import sys
from scipy.io.wavfile import write as write_wav

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

def message_to_binary_vector(message: str, latent_dim: int) -> torch.Tensor:
    """Converts a string message into a binary latent vector."""
    length_prefix_bits = 8
    max_message_chars = (latent_dim - length_prefix_bits) // 8
    if len(message) > max_message_chars:
        raise ValueError(f"Message is too long for one chunk. Max length: {max_message_chars} chars.")

    length_binary = format(len(message), f'0{length_prefix_bits}b')
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    full_binary_string = length_binary + binary_message
    binary_values = [1.0 if bit == '1' else -1.0 for bit in full_binary_string]
    padding_size = latent_dim - len(binary_values)
    padded_vector = binary_values + [0.0] * padding_size
    return torch.tensor(padded_vector, dtype=torch.float32).unsqueeze(0)

def binary_vector_to_message(vector: torch.Tensor) -> str:
    """Converts a binary latent vector back into a string message."""
    binary_string = ''.join(['1' if val > 0 else '0' for val in vector.squeeze()])
    length_prefix_bits = 8
    if len(binary_string) < length_prefix_bits:
        return "[Error: Incomplete vector]"
    
    length_binary = binary_string[:length_prefix_bits]
    message_length = int(length_binary, 2)
    message_binary = binary_string[length_prefix_bits : length_prefix_bits + message_length * 8]
    byte_chunks = [message_binary[i:i+8] for i in range(0, len(message_binary), 8)]
    message = ""
    for byte in byte_chunks:
        if len(byte) == 8:
            message += chr(int(byte, 2))
    return message

# --- Gradio Interface Functions ---

def embed_message(message: str):
    """Gradio function to embed a message and return the audio file path for multiple components."""
    if not message:
        raise gr.Error("Message cannot be empty.")

    length_prefix_bits = 8
    max_chars_per_chunk = (LATENT_DIM - length_prefix_bits) // 8
    message_chunks = [message[i:i + max_chars_per_chunk] for i in range(0, len(message), max_chars_per_chunk)]
    
    all_audio_chunks = []
    
    for chunk in message_chunks:
        try:
            latent_vector = message_to_binary_vector(chunk, LATENT_DIM).to(DEVICE)
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

def extract_message(audio_filepath):
    """Gradio function to extract a message from an uploaded audio file."""
    if audio_filepath is None:
        raise gr.Error("Please upload an audio file.")

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

    full_message = ""
    for i in range(num_chunks):
        chunk_audio = received_audio[i * AUDIO_LENGTH_SAMPLES : (i + 1) * AUDIO_LENGTH_SAMPLES]
        received_audio_tensor = torch.from_numpy(chunk_audio).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            extracted_vector = extractor(received_audio_tensor)

        decoded_chunk = binary_vector_to_message(extracted_vector)
        full_message += decoded_chunk

    return full_message if full_message else "[No message found]"

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
                embed_button = gr.Button("Generate Audio", variant="primary")
            with gr.Column():
                embed_output_audio = gr.Audio(label="Generated Audio with Hidden Message", type="filepath", show_download_button=True)
                download_file = gr.File(label="Download Audio File", visible=False)
        embed_button.click(
            fn=embed_message,
            inputs=embed_input,
            # The function returns the same path to both the audio player and the file download component
            outputs=[embed_output_audio, download_file]
        )

    with gr.Tab("Extract Message"):
        with gr.Row():
            with gr.Column():
                extract_input_audio = gr.Audio(label="Upload Audio File", type="filepath")
                extract_button = gr.Button("Extract Message", variant="primary")
            with gr.Column():
                extract_output_text = gr.Textbox(label="Extracted Message")
        extract_button.click(
            fn=extract_message,
            inputs=extract_input_audio,
            outputs=extract_output_text
        )

if __name__ == "__main__":
    # To make the UI accessible on your local network, set share=True
    demo.launch()