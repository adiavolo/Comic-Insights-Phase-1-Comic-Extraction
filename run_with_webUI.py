import gradio as gr
from PIL import Image
import numpy as np
from transformers import AutoModel
import torch
import os
import ollama  # Import Ollama library

# Detect device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the local MagiV2 model
model_path = "C:/X_Comic_Insights/magiv2"  # Adjust the path if needed
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=True).to(device).eval()
print("Model loaded successfully!")

# Function to process uploaded images


def read_image(image):
    image = Image.open(image).convert("L").convert("RGB")
    return np.array(image)

# Function to process manga and generate transcript


def process_images(chapter_pages, character_bank_images, character_bank_names):
    if not chapter_pages:
        return [], "Error: Please upload manga pages!", ""

    if not character_bank_images:
        character_bank_images = []
        character_bank_names = ""

    if not character_bank_names:
        character_bank_names = ",".join([os.path.splitext(os.path.basename(image.name))[
                                        0] for image in character_bank_images])

    chapter_pages = [read_image(image) for image in chapter_pages]
    character_bank = {
        "images": [read_image(image) for image in character_bank_images],
        "names": character_bank_names.split(",")
    }

    with torch.no_grad():
        per_page_results = model.do_chapter_wide_prediction(
            chapter_pages, character_bank, use_tqdm=True, do_ocr=True
        )

    output_images = []
    transcript = []
    for i, (image, page_result) in enumerate(zip(chapter_pages, per_page_results)):
        output_image = model.visualise_single_image_prediction(
            image, page_result, filename=None)
        output_images.append(output_image)

        speaker_name = {
            text_idx: page_result["character_names"][char_idx] for text_idx, char_idx in page_result["text_character_associations"]
        }

        for j in range(len(page_result["ocr"])):
            if not page_result["is_essential_text"][j]:
                continue
            name = speaker_name.get(j, "unsure")
            transcript.append(f"<{name}>: {page_result['ocr'][j]}")

    transcript_text = "\n".join(transcript)

    # Generate summary using DeepSeek R1 via Ollama
    summary_text = summarize_transcript(transcript_text)

    return output_images, transcript_text, summary_text

# Function to send transcript to DeepSeek R1 via Ollama


def summarize_transcript(transcript):
    prompt = f"Give the summary of the following manga transcript after carefully going through the dialogues:\n\n{transcript}"
    response = ollama.chat(model="deepseek-r1:14b", messages=[
                           {"role": "user", "content": prompt}])
    return response['message']['content']


# Define Gradio interface
chapter_pages_input = gr.Files(label="Chapter pages in chronological order.")
character_bank_images_input = gr.Files(
    label="Character reference images. If left empty, the transcript will say 'Other' for all characters.")
character_bank_names_input = gr.Textbox(
    label="Character names (comma separated). If left empty, the filenames of character images will be used.")

output_images = gr.Gallery(label="Output Images")
transcript_output = gr.Textbox(label="Transcript")
summary_output = gr.Textbox(
    label="Summary of Transcript (Generated by DeepSeek R1)")

gr.Interface(
    fn=process_images,
    inputs=[chapter_pages_input, character_bank_images_input,
            character_bank_names_input],
    outputs=[output_images, transcript_output, summary_output],
    title="Comic Insights Phase 1: Extraction",
    description="Upload manga pages and character references to generate annotated pages, transcriptions, and summaries using DeepSeek R1 via Ollama."
).launch()
