import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the Python path to ensure correct module resolution if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_training_log(log_path, save_path=None):
    """
    Reads a training log CSV file and plots the loss curves.

    Args:
        log_path (str): The full path to the training_log.csv file.
        save_path (str, optional): Path to save the plot image. If None, the plot is displayed directly.
    """
    # --- 1. Load the Data ---
    try:
        df = pd.read_csv(log_path)
        print(f"✅ Successfully loaded log file from: {log_path}")
    except FileNotFoundError:
        print(f"❌ Error: Log file not found at '{log_path}'.")
        print("Please ensure you have run the training script (`models/train.py`) to generate the log.")
        return

    # --- 2. Create the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each loss curve with distinct styles
    ax.plot(df['epoch'], df['loss_D'], label='Discriminator Loss', color='red', linestyle='-')
    ax.plot(df['epoch'], df['loss_G_GAN'], label='Generator GAN Loss', color='blue', linestyle='--')
    ax.plot(df['epoch'], df['loss_G_Recon'], label='Generator Reconstruction Loss', color='green', linestyle=':')

    # --- 3. Customize the Plot ---
    ax.set_title('Training Loss Curves', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set y-axis to start from 0 for better perspective, if all losses are positive
    if (df[['loss_D', 'loss_G_GAN', 'loss_G_Recon']] >= 0).all().all():
        ax.set_ylim(bottom=0)

    print("📊 Plot generated.")

    # --- 4. Save or Show the Plot ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Define the path to the log file based on the project structure
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_path = os.path.join(project_root, 'models', 'saved_models', 'training_log.csv')
    
    # Define where to save the output plot
    output_plot_path = os.path.join(project_root, 'research', 'training_loss_curves.png')

    # Generate and save the plot
    plot_training_log(log_file_path, save_path=output_plot_path)