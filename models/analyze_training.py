import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import librosa.display

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main evaluation function from calculate_metrics
# We will refactor calculate_metrics to make this possible
from calculate_metrics import main as run_evaluation
from models.generator import Generator
from utils.audio_tools import get_stft

# --- Configuration ---
MODEL_SAVE_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models"
GEN_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "generator_final.pth")
LATENT_DIM = 256
SAMPLE_RATE = 16000
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1
REPORT_SAVE_PATH = r"d:\EchoCrypt\EchoCrypt\training_report"
LOG_FILE = os.path.join(MODEL_SAVE_PATH, "training_log.csv")

def plot_loss_curves(log_df: pd.DataFrame, save_path: str):
    """
    Plots the training loss curves from the log data and saves the plot.
    """
    print("📊 Plotting loss curves...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plotting each loss
    ax.plot(log_df['epoch'], log_df['loss_D'], label='Discriminator Loss', color='red', alpha=0.8)
    ax.plot(log_df['epoch'], log_df['loss_G_GAN'], label='Generator GAN Loss', color='blue', alpha=0.8)
    ax.plot(log_df['epoch'], log_df['loss_G_Recon'], label='Generator Recon Loss', color='green', alpha=0.8)

    # Setting titles and labels
    ax.set_title('Training Loss Curves', fontsize=16, weight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale('log') # Use a log scale to see the details of smaller losses
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the figure
    plt.savefig(save_path)
    print(f"✅ Loss plot saved to {save_path}")
    plt.close()

def save_report(metrics: dict, report_path: str):
    """
    Saves the final evaluation metrics to a text file.
    """
    print("📝 Saving final metrics report...")
    with open(report_path, 'w') as f:
        f.write("--- EchoCrypt Training & Evaluation Report ---\n\n")
        f.write("Final Model Performance (End-to-End with FEC):\n")
        f.write("-------------------------------------------------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("-------------------------------------------------\n")
    print(f"✅ Report saved to {report_path}")

def generate_visual_comparisons(generator: Generator, save_path: str):
    """
    Generates and plots a comparison of waveforms and spectrograms for
    a standard noise audio and an audio with an embedded message.
    """
    print("🖼️ Generating visual comparison plots...")
    device = next(generator.parameters()).device
    generator.eval()

    with torch.no_grad():
        # 1. Generate standard "noise" audio from a random vector
        noise_vec = torch.randn(1, LATENT_DIM).to(device)
        noise_audio = generator(noise_vec).squeeze().cpu().numpy()

        # 2. Generate "stego" audio from a realistic, structured binary message vector
        # This MUST match the logic in train.py and app.py
        max_bytes = (LATENT_DIM - 8) // 8
        payload_len = max_bytes - 1 # Use a nearly full payload for a good visual
        payload = os.urandom(payload_len)
        len_binary = format(len(payload), '08b')
        payload_binary = ''.join(format(byte, '08b') for byte in payload)
        full_binary = [1.0 if bit == '1' else -1.0 for bit in (len_binary + payload_binary)]
        padded_vector = full_binary + [0.0] * (LATENT_DIM - len(full_binary))
        stego_vec = torch.tensor(padded_vector, dtype=torch.float32, device=device).unsqueeze(0)
        stego_audio = generator(stego_vec).squeeze().cpu().numpy()

    # 3. Compute STFTs for both
    noise_stft_mag = torch.abs(get_stft(torch.from_numpy(noise_audio).unsqueeze(0)).squeeze()).numpy()
    stego_stft_mag = torch.abs(get_stft(torch.from_numpy(stego_audio).unsqueeze(0)).squeeze()).numpy()

    # 4. Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Visual Comparison: Standard Noise vs. Steganographic Audio', fontsize=20, weight='bold')

    # Waveforms
    librosa.display.waveshow(noise_audio, sr=SAMPLE_RATE, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Standard Noise Waveform', fontsize=14)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')

    librosa.display.waveshow(stego_audio, sr=SAMPLE_RATE, ax=axes[0, 1], color='red')
    axes[0, 1].set_title('Steganographic Audio Waveform', fontsize=14)
    axes[0, 1].set_xlabel('Time (s)')

    # Spectrograms (Log-frequency scale)
    # We use the magnitude of the complex STFT result
    noise_db = librosa.amplitude_to_db(np.abs(noise_stft_mag[:,:,0] + 1j*noise_stft_mag[:,:,1]), ref=np.max)
    img1 = librosa.display.specshow(noise_db, sr=SAMPLE_RATE, x_axis='time', y_axis='log', ax=axes[1, 0])
    axes[1, 0].set_title('Standard Noise Spectrogram', fontsize=14)
    fig.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')

    stego_db = librosa.amplitude_to_db(np.abs(stego_stft_mag[:,:,0] + 1j*stego_stft_mag[:,:,1]), ref=np.max)
    img2 = librosa.display.specshow(stego_db, sr=SAMPLE_RATE, x_axis='time', y_axis='log', ax=axes[1, 1])
    axes[1, 1].set_title('Steganographic Audio Spectrogram', fontsize=14)
    fig.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    # Save the figure
    plt.savefig(save_path)
    print(f"✅ Visual comparison plot saved to {save_path}")
    plt.close()



def main():
    """
    Main function to run the analysis pipeline.
    """
    print("--- Starting Post-Training Analysis ---")
    os.makedirs(REPORT_SAVE_PATH, exist_ok=True)

    # 1. Check if log file exists
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: Training log file not found at {LOG_FILE}")
        print("Please run the training script first.")
        return

    # 2. Plot loss curves
    try:
        log_df = pd.read_csv(LOG_FILE)
        plot_save_path = os.path.join(REPORT_SAVE_PATH, "loss_curves.png")
        plot_loss_curves(log_df, plot_save_path)
    except Exception as e:
        print(f"❌ Failed to generate loss plot: {e}")

    # 3. Generate visual comparisons
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = Generator(latent_dim=LATENT_DIM)
        generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=device, weights_only=True))
        generator.to(device)

        visual_plot_path = os.path.join(REPORT_SAVE_PATH, "visual_comparison.png")
        generate_visual_comparisons(generator, visual_plot_path)
    except Exception as e:
        print(f"❌ Failed to generate visual comparisons: {e}")

    # 4. Run final evaluation and get metrics
    print("\n🚀 Running final model evaluation...")
    try:
        # We will modify calculate_metrics.py's main to return the metrics
        final_metrics = run_evaluation()
        if final_metrics:
            report_file_path = os.path.join(REPORT_SAVE_PATH, "final_metrics_report.txt")
            save_report(final_metrics, report_file_path)
        else:
            print("⚠️ Evaluation did not return metrics.")

    except Exception as e:
        print(f"❌ Failed to run evaluation: {e}")

    print("\n--- Analysis Complete ---")
    print(f"Find the report and plots in: {REPORT_SAVE_PATH}")


if __name__ == "__main__":
    main()