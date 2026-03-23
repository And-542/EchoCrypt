import sys
import os
import torch
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import Generator

def generate_and_save_audio(model, latent_dim, output_path, sample_rate=16000):
    """
    Generates audio using the generator model and saves it to a file.

    Args:
        model (nn.Module): The generator model.
        latent_dim (int): The dimension of the latent space.
        output_path (str): Path to save the generated .wav file.
        sample_rate (int): The sample rate of the audio.
    """
    # Determine device and move model and tensor
    device = next(model.parameters()).device

    
    model.eval()

    # Generate a random latent vector (this will eventually be our encoded message)
    
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)  # Batch size of 1, on the correct device
        generated_waveform = model(z)

    # Convert tensor to numpy array for saving
    audio_data = generated_waveform.squeeze().cpu().numpy()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_data, samplerate=sample_rate)
    print(f"✅ Audio generated and saved to {output_path}")

if __name__ == "__main__":
    LATENT_DIM = 100
    # --- Load the trained model ---
    MODEL_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models\generator_final.pth"
    OUTPUT_FILE = r'd:\EchoCrypt\EchoCrypt\data\noise\trained_generated_noise.wav'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate the generator
    generator = Generator(latent_dim=LATENT_DIM)

    # Load the saved weights from training
    print(f"💿 Loading trained model from {MODEL_PATH}")
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.to(DEVICE)

    # Generate and save the audio
    generate_and_save_audio(generator, LATENT_DIM, OUTPUT_FILE)
