import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
import csv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from models.generator import Generator
from models.discriminator import Discriminator
from models.extractor import Extractor # Import the Extractor
from utils.audio_tools import get_stft

# --- Hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 64
LR_GEN = 0.0002  # Learning rate for the generator
LR_DISC = 0.00005 # Slower learning rate for the discriminator
BETA1 = 0.5  # Adam optimizer parameter
LAMBDA_RECON = 2.0 # Weight for the reconstruction loss
LATENT_DIM = 100
G_UPDATES_PER_D = 2 # Update generator twice for every discriminator update
NUM_BATCHES_PER_EPOCH = 1000 // BATCH_SIZE # For demonstration purposes
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
    ext = Extractor(input_length=AUDIO_LENGTH_SAMPLES, latent_dim=LATENT_DIM).to(DEVICE)

    # --- Optimizers & Loss ---
    opt_gen = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR_DISC, betas=(BETA1, 0.999))
    opt_ext = optim.Adam(ext.parameters(), lr=LR_GEN, betas=(BETA1, 0.999)) # Extractor can share LR with Gen
    criterion_gan = nn.BCEWithLogitsLoss() # For GAN loss
    criterion_recon = nn.MSELoss() # For reconstruction loss

    # --- Logging Setup ---
    log_file_path = os.path.join(MODEL_SAVE_PATH, "training_log.csv")
    log_history = []

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        epoch_loss_d = 0.0
        epoch_loss_g_gan = 0.0
        epoch_loss_g_recon = 0.0

        # Using tqdm for a nice progress bar
        loop = tqdm(range(NUM_BATCHES_PER_EPOCH), desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for batch_idx in loop:
            # --- Train Discriminator ---
            disc.zero_grad()

            # 1. Train with REAL audio
            real_audio = torch.randn(BATCH_SIZE, AUDIO_LENGTH_SAMPLES).to(DEVICE)
            real_stft = get_stft(real_audio)
            disc_real = disc(real_stft).view(-1)
            loss_disc_real = criterion_gan(disc_real, torch.full_like(disc_real, 0.9))
            loss_disc_real.backward()

            # 2. Train with FAKE audio
            latent_vec = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            fake_audio = gen(latent_vec)
            fake_stft = get_stft(fake_audio.detach())
            disc_fake = disc(fake_stft).view(-1)
            loss_disc_fake = criterion_gan(disc_fake, torch.zeros_like(disc_fake))
            loss_disc_fake.backward()

            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            opt_disc.step()

            # --- Train Generator (multiple times) ---
            for _ in range(G_UPDATES_PER_D):
                gen.zero_grad()
                ext.zero_grad()

                # Generate a new batch of fake audio for each generator update
                # For the GAN loss, we use a standard random vector to ensure variety
                latent_vec_gan = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
                fake_audio_g = gen(latent_vec_gan)

                # For the RECONSTRUCTION loss, we MUST use the same kind of binary vectors
                # as our message embedder. This is the key to closing the domain gap.
                binary_vec_recon = (torch.randint(0, 3, (BATCH_SIZE, LATENT_DIM), device=DEVICE) - 1).float() # Creates a tensor of -1, 0, 1
                fake_audio_recon = gen(binary_vec_recon)

                # --- Calculate GAN Loss for Generator ---
                # We want the generator to produce audio that the discriminator thinks is REAL (label 1)
                output = disc(get_stft(fake_audio_g)).view(-1)
                loss_gan_gen = criterion_gan(output, torch.ones_like(output))

                # --- Calculate Reconstruction Loss for Autoencoder ---
                # We want the extractor to reconstruct the original latent vector from the fake audio
                reconstructed_vec = ext(fake_audio_recon)
                loss_recon = criterion_recon(reconstructed_vec, binary_vec_recon)

                # --- Combined Loss ---
                # The generator is updated by both losses. The extractor is only updated by the reconstruction loss.
                # Backpropagating this combined loss updates both G and E based on their respective contributions.
                loss_gen_combined = loss_gan_gen + LAMBDA_RECON * loss_recon
                loss_gen_combined.backward()

                opt_gen.step()
                opt_ext.step()

            # Accumulate losses for epoch average
            epoch_loss_d += loss_disc.item()
            epoch_loss_g_gan += loss_gan_gen.item()
            epoch_loss_g_recon += loss_recon.item()

            # Update progress bar
            loop.set_postfix(
                loss_D=f"{loss_disc.item():.4f}",
                loss_G_GAN=f"{loss_gan_gen.item():.4f}",
                loss_G_Recon=f"{loss_recon.item():.4f}"
            )

        # --- Log Epoch Results ---
        avg_loss_d = epoch_loss_d / NUM_BATCHES_PER_EPOCH
        avg_loss_g_gan = epoch_loss_g_gan / NUM_BATCHES_PER_EPOCH
        avg_loss_g_recon = epoch_loss_g_recon / NUM_BATCHES_PER_EPOCH
        log_entry = {
            "epoch": epoch + 1,
            "loss_D": avg_loss_d,
            "loss_G_GAN": avg_loss_g_gan,
            "loss_G_Recon": avg_loss_g_recon
        }
        log_history.append(log_entry)

        # --- Save Models ---
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_path_gen = os.path.join(MODEL_SAVE_PATH, f"generator_epoch_{epoch+1}.pth")
            save_path_disc = os.path.join(MODEL_SAVE_PATH, f"discriminator_epoch_{epoch+1}.pth")
            save_path_ext = os.path.join(MODEL_SAVE_PATH, f"extractor_epoch_{epoch+1}.pth")
            torch.save(gen.state_dict(), save_path_gen)
            torch.save(disc.state_dict(), save_path_disc)
            torch.save(ext.state_dict(), save_path_ext)
            print(f"\n💾 Models saved at epoch {epoch+1}")

    print("✅ Training complete.")
    # Save final models
    torch.save(gen.state_dict(), os.path.join(MODEL_SAVE_PATH, "generator_final.pth"))
    torch.save(disc.state_dict(), os.path.join(MODEL_SAVE_PATH, "discriminator_final.pth"))
    torch.save(ext.state_dict(), os.path.join(MODEL_SAVE_PATH, "extractor_final.pth"))
    print("💾 Final models saved.")

    # --- Save Log File ---
    print(f"📝 Saving training log to {log_file_path}")
    with open(log_file_path, 'w', newline='') as csvfile:
        fieldnames = ["epoch", "loss_D", "loss_G_GAN", "loss_G_Recon"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_history)
    print("✅ Log file saved.")


if __name__ == "__main__":
    main()