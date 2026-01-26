import matplotlib.pyplot as plt
import csv
import argparse
import os

def plot_training_loss(log_file, save_plot=True):
    """
    Reads a CSV training log and plots the loss curves.
    Returns the matplotlib figure object.
    """
    if not os.path.exists(log_file):
        print(f"❌ Error: File '{log_file}' not found.")
        return None

    print(f"📊 Reading training log: {log_file}")

    epochs = []
    losses = {} 

    try:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            
            # Normalize headers to handle whitespace
            if reader.fieldnames:
                reader.fieldnames = [h.strip() for h in reader.fieldnames]
            else:
                print("❌ Error: CSV file appears to be empty or invalid.")
                return None

            headers = reader.fieldnames
            
            # Identify columns to plot (containing 'loss')
            plot_keys = [h for h in headers if 'loss' in h.lower()]
            
            if not plot_keys:
                print("⚠️ No columns containing 'loss' found. Plotting all numeric columns except 'epoch'.")
                plot_keys = [h for h in headers if 'epoch' not in h.lower()]

            for k in plot_keys:
                losses[k] = []

            for i, row in enumerate(reader):
                # Parse Epoch
                try:
                    if 'epoch' in row:
                        e = float(row['epoch'])
                    elif 'Epoch' in row:
                        e = float(row['Epoch'])
                    else:
                        e = i + 1
                    epochs.append(e)
                except ValueError:
                    continue 

                # Parse Values
                for k in plot_keys:
                    try:
                        val = float(row[k])
                        losses[k].append(val)
                    except ValueError:
                        losses[k].append(None)

        # Plotting
        fig = plt.figure(figsize=(10, 6))
        
        has_data = False
        for name, values in losses.items():
            clean_epochs = [e for e, v in zip(epochs, values) if v is not None]
            clean_values = [v for v in values if v is not None]
            
            if clean_values:
                plt.plot(clean_epochs, clean_values, label=name)
                has_data = True

        if not has_data:
            print("❌ No valid data to plot.")
            return None

        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_plot:
            output_path = os.path.splitext(log_file)[0] + "_plot.png"
            plt.savefig(output_path, dpi=300)
            print(f"✅ Graph saved to: {output_path}")
            
        return fig

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training loss from a CSV log file.")
    parser.add_argument(
        "log_file", 
        type=str, 
        nargs='?', 
        default=r"D:\EchoCrypt\EchoCrypt\models\saved_models\training_log.csv", 
        help="Path to the training log CSV file."
    )
    args = parser.parse_args()
    
    plot_training_loss(args.log_file)