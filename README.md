# **Comic Insights: Phase 1 - Comic Strip Context Extraction**

## **Overview**
This project builds upon **MagiV2** by **Ragav Sachdeva et al.** and extends its functionality by integrating **DeepSeek R1** for AI-powered summarization. Phase 1 focuses on **extracting and summarizing textual content** from comic strips using **OCR (Tesseract)** and **AI-based summarization models**.

## **Features**
- **Comic Page Upload:** Accepts JPEG/PNG images of comic pages.
- **OCR Text Extraction:** Uses **Tesseract OCR** to extract text from speech bubbles and captions.
- **AI-Powered Summarization:** Utilizes **DeepSeek R1** (via Ollama) to generate concise summaries.
- **Interactive UI:** Built with **Gradio** for an easy-to-use interface.

## **Acknowledgments**
This project is built on the **MagiV2** framework, developed by **Ragav Sachdeva et al.**. The original repository can be found here:

🔗 [MagiV2 Repository](https://github.com/ragavsachdeva/magi)

The model and datasets from MagiV2 are used under the **academic research license** and are not intended for commercial use. Please ensure proper citation when using this work.

## **Installation & Setup**
### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- Tesseract OCR (`pytesseract`)
- Ollama (for DeepSeek R1 integration)
- PyTorch with CUDA (for GPU acceleration)

### **Setup Instructions**
```bash
# Clone this repository
git clone https://github.com/yourusername/comic-insights.git
cd comic-insights

# Install dependencies
pip install -r requirements.txt

# Ensure Tesseract OCR is installed and configured
sudo apt install tesseract-ocr  # Linux
brew install tesseract           # MacOS
choco install tesseract          # Windows

# Run the Gradio interface
python your_script.py
```

## **Usage**
1. **Upload comic images** via the Gradio web interface.
2. **Extracted text** will be displayed after OCR processing.
3. **AI-generated summaries** will appear after processing with DeepSeek R1.
4. **Download or copy the transcript and summary** for further use.

## **Citation**
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

## **License**
This project is licensed for **academic research purposes only** in accordance with **MagiV2's licensing terms**. Commercial use and redistribution of the model files are strictly prohibited.

---
🚀 **Future Work:** Phase 2 will focus on **AI-generated comic panels** using NLP and generative models like **Stable Diffusion**.
