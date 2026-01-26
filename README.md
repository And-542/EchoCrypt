
🎧 EchoCrypt – Coverless Audio Steganography Using GANs

## 🚀 Project Overview
EchoCrypt is a research-grade **crypto-steganography** system. It hides secret messages within GAN-generated audio and secures them with **AES encryption**. This two-layer approach ensures that even if the hidden audio is detected, the message remains unreadable without the correct password.
It is a **coverless** method, meaning it generates new audio that already contains the encrypted data, rather than modifying an existing cover file.

---

## 🗂️ Project Structure

```bash
EchoCrypt/
│
├── data/
│   ├── noise/              # Generated white noise samples
│   └── binary/             # Binary files to be embedded
│
├── models/
│   ├── generator.py         # Generator model
│   ├── discriminator.py     # 🧠 Discriminator model (for training the GAN)
│   ├── extractor.py         # 🔍 Decoder/extractor model (optional, if you're using a CNN)
│   ├── train.py             # GAN training script (uses generator + discriminator)
│   └── saved_models/        # Stores trained models
│
│
├── utils/
│   └── audio_tools.py      # Audio pre/post-processing functions (STFT, etc.)
│
├── scripts/
│   ├── generate_noise.py   # Script to generate random noise for testing
│
│
├── research/
│   ├── stego_paper.pdf     # Reference research papers
│   └── project_diagram.png # Architecture/flowchart
│
├── README.md
├── .gitignore
├── requirements.txt
├── calculate_metrics.py    # Script to calculate BER, MER, and SNR
└── main.py                 # Optional central execution point
🧠 Technologies & Methods
Coverless Audio Steganography
**AES-256 (CBC mode)** for strong symmetric encryption
GAN (Generative Adversarial Networks) for generating white noise that encodes binary data
Differentiable STFT using PyTorch
Optional CNN-based decoder for data extraction

Tools: Python, PyTorch, Gradio, Librosa, NumPy, SoundFile, PyCryptodome, TQDM

⚙️ Setup Instructions
Recommended: Use Anaconda and Python 3.10+

bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-username/echocrypt.git
cd echocrypt

# Create environment
conda create -n echocrypt python=3.10 -y
conda activate echocrypt

# Install dependencies
pip install -r requirements.txt
🧪 Usage Example
The easiest way to use EchoCrypt is via the Gradio web interface.

```bash
# Launch the web UI
python app.py
```
👩‍💻 Contributors
Andrew Varghese Koshy – GAN Engineering
Jenit Mathew – Data Processing, Testing, and Research

📚 Research Motivation
This project explores a modern approach to steganography by leveraging generative models instead of modifying existing audio files. It aims to build a robust, undetectable, and efficient data-hiding mechanism for secure communication.

🛡️ Disclaimer
This project is for educational and ethical use only. Do not use it to transmit illegal or unauthorized content.

📫 Contact
Open an issue on GitHub or reach out at [your-email@example.com].

⭐ Star This Repo
If you find this helpful or interesting, feel free to ⭐ star it and share with others!                                                                                                                                                                               
