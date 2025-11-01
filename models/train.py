import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from models.generator import Generator
from models.discriminator import Discriminator
from utils.audio_tools import get_stft

# --- Hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 64
LR_GEN = 0.0002  # Learning rate for the generator
LR_DISC = 0.00005 # Slower learning rate for the discriminator
BETA1 = 0.5  # Adam optimizer parameter
LATENT_DIM = 100
G_UPDATES_PER_D = 2 # Update generator twice for every discriminator update
SAMPLE_RATE = 16000
AUDIO_LENGTH_SECONDS = 1
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * AUDIO_LENGTH_SECONDS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models"
SAVE_INTERVAL = 10  # Save models every 10 epochs

def main():
    """Main training loop for the GAN."""
    # --- Device Check ---
    print("--------------------")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Selected device: {DEVICE}")
    print("--------------------")

    # --- Setup ---
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # --- Models ---
    # Note: The Generator is designed to output a raw waveform of a specific length.
    gen = Generator(latent_dim=LATENT_DIM, output_length=AUDIO_LENGTH_SAMPLES).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    # --- Optimizers & Loss ---
    opt_gen = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR_DISC, betas=(BETA1, 0.999))
    criterion = nn.BCEWithLogitsLoss() # More stable than BCELoss and Sigmoid

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        # Using tqdm for a nice progress bar
        loop = tqdm(range(0, 1000, BATCH_SIZE), desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for _ in loop:
            # --- Train Discriminator ---
            disc.zero_grad()

            # 1. Train with REAL audio
            real_audio = torch.randn(BATCH_SIZE, AUDIO_LENGTH_SAMPLES).to(DEVICE)
            real_stft = get_stft(real_audio)
            disc_real = disc(real_stft).view(-1)
            loss_disc_real = criterion(disc_real, torch.full_like(disc_real, 0.9))
            loss_disc_real.backward()

            # 2. Train with FAKE audio
            latent_vec = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            fake_audio = gen(latent_vec)
            fake_stft = get_stft(fake_audio.detach())
            disc_fake = disc(fake_stft).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc_fake.backward()

            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            opt_disc.step()

            # --- Train Generator (multiple times) ---
            for _ in range(G_UPDATES_PER_D):
                gen.zero_grad()

                # Generate a new batch of fake audio for each generator update
                latent_vec_g = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
                fake_audio_g = gen(latent_vec_g)

                # We want the generator to produce audio that the discriminator thinks is REAL
                output = disc(get_stft(fake_audio_g)).view(-1)
                loss_gen = criterion(output, torch.ones_like(output)) # Fool discriminator with label 1
                loss_gen.backward()
                opt_gen.step()

            # Update progress bar
            loop.set_postfix(
                loss_D=f"{loss_disc.item():.4f}",
                loss_G=f"{loss_gen.item():.4f}"
            )

        # --- Save Models ---
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_path_gen = os.path.join(MODEL_SAVE_PATH, f"generator_epoch_{epoch+1}.pth")
            save_path_disc = os.path.join(MODEL_SAVE_PATH, f"discriminator_epoch_{epoch+1}.pth")
            torch.save(gen.state_dict(), save_path_gen)
            torch.save(disc.state_dict(), save_path_disc)
            print(f"\n💾 Models saved at epoch {epoch+1}")

    print("✅ Training complete.")
    # Save final models
    torch.save(gen.state_dict(), os.path.join(MODEL_SAVE_PATH, "generator_final.pth"))
    torch.save(disc.state_dict(), os.path.join(MODEL_SAVE_PATH, "discriminator_final.pth"))
    print("💾 Final models saved.")


if __name__ == "__main__":
    main()