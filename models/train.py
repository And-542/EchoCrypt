import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# Import our custom modules
from models.generator import Generator
from models.discriminator import Discriminator
from utils.audio_tools import get_stft

# --- Hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.0002
BETA1 = 0.5  # Adam optimizer parameter
LATENT_DIM = 100
SAMPLE_RATE = 16000
AUDIO_LENGTH_SECONDS = 1
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * AUDIO_LENGTH_SECONDS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = r"d:\EchoCrypt\EchoCrypt\models\saved_models"
SAVE_INTERVAL = 10  # Save models every 10 epochs

def main():
    """Main training loop for the GAN."""
    print(f"🚀 Starting training on {DEVICE}...")

    # --- Setup ---
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # --- Models ---
    # Note: The Generator is designed to output a raw waveform of a specific length.
    gen = Generator(latent_dim=LATENT_DIM, output_length=AUDIO_LENGTH_SAMPLES).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    # --- Optimizers & Loss ---
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(BETA1, 0.999))
    criterion = nn.BCEWithLogitsLoss() # More stable than BCELoss and Sigmoid

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        # Using tqdm for a nice progress bar
        loop = tqdm(range(0, 1000, BATCH_SIZE), desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for _ in loop:
            ### Train Discriminator ###
            disc.zero_grad()

            # 1. Train with REAL audio
            # We generate random noise on the fly to act as our "real" data
            real_audio = torch.randn(BATCH_SIZE, AUDIO_LENGTH_SAMPLES).to(DEVICE)
            real_stft = get_stft(real_audio)
            
            disc_real = disc(real_stft).view(-1)
            # Label smoothing: use 0.9 for real labels instead of 1.0
            loss_disc_real = criterion(disc_real, torch.full_like(disc_real, 0.9))
            loss_disc_real.backward()

            # 2. Train with FAKE audio
            latent_vec = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            fake_audio = gen(latent_vec)
            fake_stft = get_stft(fake_audio.detach()) # detach to avoid backprop into generator
            disc_fake = disc(fake_stft).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # Label as 0 (fake)
            loss_disc_fake.backward()

            # Update discriminator weights
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            opt_disc.step()

            ### Train Generator ###
            gen.zero_grad()

            # We want the generator to produce audio that the discriminator thinks is REAL
            # We use the non-detached fake_audio tensor here
            output = disc(get_stft(fake_audio)).view(-1)
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