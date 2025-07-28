
🎧 EchoCrypt – Coverless Audio Steganography Using GANs

## 🚀 Project Overview

EchoCrypt is a research-grade audio steganography system that uses **GAN-generated white noise** to hide binary data without modifying existing files — making it stealthy and virtually undetectable. It is a **coverless** method, meaning it generates audio that already contains the embedded data, unlike traditional stego methods that modify original media.

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
│   ├── generator.py        # GAN noise generator
│   ├── train.py            # Training logic
│   └── saved_models/       # Saved weights and checkpoints
│
├── utils/
│   └── audio_tools.py      # Audio pre/post-processing functions (STFT, etc.)
│
├── scripts/
│   ├── generate_noise.py   # Script to generate stego noise
│   ├── embed_data.py       # Script to embed binary into audio
│   └── extract_data.py     # Script to extract data from audio
│
├── research/
│   ├── stego_paper.pdf     # Reference research papers
│   └── project_diagram.png # Architecture/flowchart
│
├── README.md
├── .gitignore
├── requirements.txt
└── main.py                 # Optional central execution point
🧠 Technologies & Methods
Coverless Audio Steganography

GAN (Generative Adversarial Networks) for generating white noise that encodes binary data

Differentiable STFT/iSTFT using Librosa

Optional CNN-based decoder for data extraction

Tools: Python, PyTorch, Librosa, NumPy, SoundFile, TQDM, VS Code, GitHub Desktop

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
bash
Copy
Edit
# Generate white noise with embedded binary data
python scripts/generate_noise.py

# Extract hidden data from audio
python scripts/extract_data.py
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
