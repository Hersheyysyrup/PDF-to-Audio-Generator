# PDF-to-Audio Document Assistant

Upload a PDF, ask questions about it, and listen to the answers. The app retrieves the most relevant passages from your document, has an LLM answer from them with page citations, and can read the answer aloud with a neural voice.

## Features

- **Scanned PDFs supported:** pages with no text layer are read with Tesseract OCR
- **Grounded answers:** retrieval-augmented generation (RAG) with page-number citations
- **Streaming answers** with automatic fallback from Gemini to Groq
- **On-demand narration** using Piper TTS
- **Caching:** processed PDFs are indexed once and reused, and generated audio is reused for identical text

## How it works

![Architecture](docs/architecture.png)

## Setup

**1. Install system dependency:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and make sure it is on your PATH.

**2. Install Python packages:**

```bash
pip install -r requirements.txt
pip install streamlit openai piper-tts
```

**3. Add API keys** in a `.env` file in the project root:

```
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
```

**4. Download the Piper voice** (`en_US-lessac-medium.onnx` and its `.onnx.json`) from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) into the `models/` folder. To use another location, set `PIPER_MODEL_PATH`.

## Run

```bash
streamlit run web/web.py
```

Upload a PDF in the sidebar, click **Load**, then ask your questions in the chat. Press **Narrate this answer** to hear a response.

## Project structure

```
document_loaders/   PDF text extraction and OCR fallback
text_splitters/     Chunking
embeddings/         BGE embedding model
vector_store/       Chroma index (one per PDF)
retriever/          MMR retriever
gen_ai/             Gemini and Groq answer generation
narration/          Piper text-to-speech
web/                Streamlit app
```

## Tech stack

Python, LangChain, PyMuPDF, Tesseract, HuggingFace `BAAI/bge-small-en-v1.5`, Chroma, Google Gemini, Groq, Piper, Streamlit.

## Notes

- Questions and retrieved passages are sent to Gemini or Groq, so an internet connection is required and document content leaves your machine.
- Never commit your `.env` file. It is already listed in `.gitignore`.
