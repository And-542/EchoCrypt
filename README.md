# EchoCrypt 🤫: Covert Communication via Generative Noise

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
![Status](https://img.shields.io/badge/status-in%20progress-orange.svg) 

## About EchoCrypt

**EchoCrypt** is an innovative deep learning project pioneering next-generation covert communication. Unlike traditional steganography that modifies existing files, EchoCrypt redefines the paradigm by **generating the cover medium (white noise audio) from scratch**, making hidden messages virtually undetectable.

At its core, EchoCrypt employs **Generative Adversarial Networks (GANs)** to synthesize seemingly innocuous white noise audio. Crucially, secret binary data is embedded directly into this audio during its very creation, ensuring no 'original' file exists for comparison, thereby maximizing stealth. To achieve robust and reliable data extraction, EchoCrypt integrates **differentiable Short-Time Fourier Transform (STFT) and Inverse STFT (iSTFT) layers** within its neural network architecture. This allows the system to operate effectively in the frequency domain, enabling the Generator to learn how to embed messages resiliently, and the Extractor to recover them accurately, even when the stego-audio undergoes common distortions like compression or the addition of environmental noise.

This project aims to demonstrate high imperceptibility (perceptually indistinguishable from genuine random static), alongside high capacity for binary data, and robust extraction capabilities against various audio attacks.

## Key Features

* **Coverless Steganography:** Generates the stego-medium (white noise) from scratch with the hidden message embedded.
* **Deep Learning-based:** Utilizes Generative Adversarial Networks (GANs) for synthetic audio generation and a robust Extractor network.
* **Differentiable Audio Transforms:** Leverages `torchaudio`'s differentiable STFT/iSTFT for learning robust embeddings and extractions in the frequency domain.
* **High Stealth:** Aims for audio output that is statistically and perceptually indistinguishable from pure white noise.
* **Robustness:** Designed to withstand common audio degradations (e.g., compression, added noise, filtering).
* **Binary Data Embedding:** Focuses on hiding arbitrary binary sequences.

## Project Structure
├── data/
│   ├── raw_noise/        # Directory for pre-generated pure white noise (training real samples)
│   └── binary_messages/  # Directory for example binary message files
├── models/
│   ├── generator.py      # PyTorch Generator network definition
│   ├── discriminator.py  # PyTorch Discriminator network definition
│   └── extractor.py      # PyTorch Extractor network definition
├── utils/
│   ├── audio_utils.py    # Helper functions for audio loading, saving, processing (e.g., STFT/iSTFT wrappers)
│   └── data_utils.py     # Helper functions for binary data manipulation, dataset creation
├── config/               # (Optional) For storing configuration files (e.g., YAML, JSON)
│   └── default_config.py
├── notebooks/
│   ├── 01_pytorch_audio_basics.ipynb  # Jupyter notebook for initial explorations
│   └── 02_gan_tutorial.ipynb          # Jupyter notebook for GAN understanding
├── train.py              # Main script for training the EchoCrypt system (G, D, E)
├── evaluate.py           # Script for evaluating stealth, capacity, and robustness
├── generate_stego.py     # Script to generate stego-audio with a specific message
├── requirements.txt      # Python dependencies
└── README.md             # This file


---

## Getting Started

Follow these instructions to set up the project locally and get a basic understanding of its components.

### Prerequisites

* Python 3.8+
* NVIDIA GPU with CUDA support (Recommended for training)
* `conda` or `venv` for environment management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/EchoCrypt.git
    cd EchoCrypt
    ```

2.  **Create and activate a Conda environment (recommended):**
    ```bash
    conda create -n echocrypt python=3.9
    conda activate echocrypt
    ```
    *Alternatively, use `python -m venv .venv` and `source .venv/bin/activate`*

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `torchaudio` requires specific CUDA versions, refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for the exact installation command)*

4.  **Verify GPU (Optional but Recommended):**
    ```python
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    ```
    You should see `True` and your GPU's name (e.g., `NVIDIA GeForce RTX 4070 Laptop GPU`).

---

## Usage

### 1. Prepare Data

Before training, you'll need a dataset of pure white noise audio samples.
* Run the data generation script:
    ```bash
    python [path/to/your/data_generation_script.py] # e.g., utils/data_utils.py or a dedicated script
    ```
    This script will generate `*.wav` files of pure Gaussian white noise in the `data/raw_noise/` directory.

### 2. Train the EchoCrypt System

To train the Generator, Discriminator, and Extractor networks:
```bash
python train.py --epochs [num_epochs] --batch_size [batch_size] --lr [learning_rate] # Add your specific args
Refer to train.py for available command-line arguments and configuration options.

Training progress and metrics will be logged (e.g., to TensorBoard).

3. Generate Stego-Audio
Once models are trained, you can generate new white noise audio with an embedded binary message:

Bash

python generate_stego.py --model_path [path_to_generator_checkpoint] --message "[your_binary_string_e.g._010110]" --output_file [output.wav]
Or, you can load a binary message from a file:

Bash

python generate_stego.py --model_path [path_to_generator_checkpoint] --message_file [path_to_binary_file.txt] --output_file [output.wav]
4. Evaluate and Extract
To evaluate the system's performance (stealth, capacity, robustness) and extract messages:

Bash

python evaluate.py --stego_audio [path_to_stego.wav] --original_message "[original_binary_string]" --extractor_path [path_to_extractor_checkpoint]
This script will output the extracted binary message and calculate metrics like Bit Error Rate (BER).

Additional arguments can be passed to simulate various audio attacks (e.g., --add_noise, --compress_mp3).


Results & Demos
(This section should be filled in as you achieve milestones. You can use images, audio links, and tables.)

Initial Prototype (Month [X]): [Description of what was achieved, e.g., "Basic GAN generating static, 10-bit message embedding working."]

Perceptual Stealth Demo:

[Link to pure generated noise sample.wav]

[Link to stego-noise sample.wav (same message, different generation)]

Can you hear the difference?

Robustness against MP3 Compression:

BER: [X]% for [Y] bits at [Z] kbps MP3 compression.

[Link to compressed stego-audio.mp3 (after re-encoding to wav for extraction)]

Capacity Achieved: Successfully embedded [X] bits per second of audio with a BER of [Y]%.

Future Work
[ ] Explore more advanced GAN architectures (e.g., StyleGAN, VQ-GAN) for improved noise quality and control.

[ ] Implement a perceptual loss function to better guide the Generator towards human-imperceptible embeddings.

[ ] Develop a more sophisticated steganalysis module to actively test the imperceptibility of the generated noise.

[ ] Experiment with different types of cover audio beyond white noise (e.g., ambient soundscapes).

[ ] Optimize for real-time embedding and extraction.

Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.

Create a new feature branch (git checkout -b feature/your-feature-name).

Commit your changes (git commit -m 'feat: Add new feature X').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

Please ensure your code adheres to PEP 8 guidelines and includes appropriate documentation (docstrings, comments).

License
This project is licensed under the MIT License - see the LICENSE file for details.

Authors
[Your Name] - [Your GitHub Profile Link] - [Your University/Affiliation]

[Partner's Name] - [Partner's GitHub Profile Link] - [Partner's University/Affiliation]

Acknowledgments
Inspired by research in generative models and differentiable signal processing, particularly works like "Hide and Speak: Towards Deep Neural Audio Steganography" ([Link to relevant paper if public])

Built with PyTorch and torchaudio.

[Any other libraries or resources that were particularly helpful]
