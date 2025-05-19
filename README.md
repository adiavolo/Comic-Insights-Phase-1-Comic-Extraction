# Comic Insights: Phase 1 - Comic Strip Context Extraction

## Overview

This project, developed for our Software Engineering course following professor’s guidance to reverse engineer comic generation, uses OCR to extract text from comic strips and deep learning NLP models for summarization. Built with PyTorch and Transformer-based architectures, it captures dialogue and story flow, producing concise summaries that aid in quick content understanding and creative analysis. 

## Features

- **Comic Page Upload:** Accepts JPEG/PNG images of comic pages.
- **OCR Text Extraction:** Uses Tesseract OCR to extract text from speech bubbles and captions.
- **AI-Powered Summarization:** Utilizes DeepSeek R1 (via Ollama) to generate concise summaries.
- **Interactive UI:** Built with Gradio for an easy-to-use interface.


## Installation & Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Tesseract OCR (`pytesseract`)
- Ollama (for DeepSeek R1 integration)
- PyTorch with CUDA (for GPU acceleration)

### Setup Instructions

```bash
# Clone this repository
git clone https://github.com/adiavolo/Comic-Insights-Phase-1-Comic-Extraction.git
cd Comic-Insights-Phase-1-Comic-Extraction

# Install dependencies
pip install -r requirements.txt

# Ensure Tesseract OCR is installed and configured
# For Linux
sudo apt install tesseract-ocr
# For macOS
brew install tesseract
# For Windows 9with chocolatey)
choco install tesseract

# Run the Gradio interface
python run_with_webUI.py
```

## Usage

1. **Upload comic images** via the Gradio web interface.
2. **Upload Character Images and Labels** will be used for OCR processing.
3. **Extracted text** will be displayed after OCR processing.
4. **AI-generated summaries** will appear after processing with DeepSeek R1.
5. **Download or copy the transcript and summary** for further use.

## Citation

If you use this project for academic purposes, please cite **MagiV2**:

```
@misc{magiv2,
  author={Ragav Sachdeva and Gyungin Shin and Andrew Zisserman},
  title={Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names},
  year={2024},
  eprint={2408.00298},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2408.00298},
}
```

## License

This project is licensed for **academic research purposes only** in accordance with **MagiV2's licensing terms**. Commercial use and redistribution of the model files are strictly prohibited.

---

🚀 **Future Work:** [Phase 2](https://github.com/adiavolo/Comic-Insights-Phase2-Comic-Generation) will focus on **AI-generated comic panels** using NLP and generative models like **Stable Diffusion**.
