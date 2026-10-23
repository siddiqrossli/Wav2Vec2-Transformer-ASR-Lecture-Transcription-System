# 🎙️ ClassScript: ASR Lecture Transcriber & AI Study Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face ZeroGPU](https://img.shields.io/badge/Hugging%20Face-ZeroGPU%20(A10G)-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/siddiqrossli/classscript-app)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg?logo=google&logoColor=white)](https://aistudio.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/UiTM-Final%20Year%20Project-7A1C8E.svg)](https://uitm.edu.my/)

**An end-to-end Automated Speech Recognition (ASR) lecture transcription engine and intelligent study companion, deployed 24/7 on free enterprise GPU hardware with an authentic bespoke UI.**

[🌐 Live Cloud Demo](https://siddiqrossli-classscript-app.hf.space) • [📦 Model Weights Hub](https://huggingface.co/siddiqrossli/asr-lecture-model) • [🚀 HF Space Repo](https://huggingface.co/spaces/siddiqrossli/classscript-app) • [📚 Systems Architecture Report](docs/SYSTEMS_ARCHITECTURE_AND_POST_MORTEM.md)

</div>

---

## 🌟 Executive Overview

**ClassScript** addresses a critical academic challenge: university students spending hours listening to fast-paced, bilingual (Malay and English) technical lectures without searchable text or organized revision tools.

The platform bridges deep learning speech recognition with modern generative AI:
1. **Acoustic Modeling Tier**: Fine-tuned **Wav2Vec 2.0** acoustic model (94.4M parameters) converting raw 16kHz audio into phoneme connectionist temporal classification (CTC) probability distributions.
2. **Language Modeling Tier**: Compiled **3-gram KenLM Language Model** integrated via CTC beam search decoding (`pyctcdecode`), rescoring candidate hypotheses with domain vocabulary.
3. **Cognitive Tier**: Integration with the **Google Gemini 2.5 Flash API** to generate automated active-recall flashcards, multiple-choice study quizzes, executive summaries, technical glossaries, and grounded contextual Q&A chat.
4. **Authentic Web Interface**: A custom-crafted HTML5/CSS3/JavaScript dashboard featuring UiTM academic branding, dark/light theme toggles, interactive waveform audio player, and **synchronized Spotify-style karaoke sentence tracking**.

---

## 🏗️ Systems Architecture & The ZeroGPU Event Queue Bridge

### The $0 Cloud Deployment Challenge
Deploying an enterprise-grade ASR pipeline normally requires expensive dedicated cloud GPUs ($350–$1,000+/mo). To deploy ClassScript for **RM0 / $0 total cost** operating 24/7 without keeping a local laptop running, the application was deployed to **Hugging Face ZeroGPU** (`zero-a10g`).

However, ZeroGPU strictly enforces a multi-tenant sandbox:
* It locks out standard Flask/WSGI servers and mandates native Gradio or Streamlit lifecycles.
* It dynamically leases an **NVIDIA A10G GPU slice** *only* when requests originate from Gradio's internal WebSocket event queue with an authenticated visitor token (`X-IP-Token`).

### The Architectural Solution: The Loopback Bridge
Rather than replacing the authentic custom frontend with a generic Gradio template, a **two-tier ASGI/WSGI Event Queue Bridge** was engineered:

```
========================================================================================
                          CLASSCRIPT ARCHITECTURE TOPOLOGY
========================================================================================

 Client Browser (Authentic Custom UI)
            │
            │ HTTPS Request (:443)
            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  FastAPI / ASGI Gateway (Port 7860)                                                │
 │                                                                                    │
 │  [ Route: / ] ──────────► a2wsgi Middleware ──► Authentic Flask Web App            │
 │  (Authentic Custom UI)                              │                              │
 │                                                     │ HTTP POST /transcribe        │
 │                                                     ▼                              │
 │                                                Extract X-IP-Token                  │
 │                                                     │                              │
 │  [ Route: /gradio_api ] ◄───────────────────────────┴ (Loopback 127.0.0.1:7860)    │
 │  Gradio WebSocket Event Queue (Endpoint: /transcribe_zerogpu)                      │
 └─────────────────────────────────┬──────────────────────────────────────────────────┘
                                   │ Dynamic GPU Lease (duration=30s)
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │  ZeroGPU Allocator (NVIDIA A10G 24GB Slice)      │
          │  1. Audio Resample & Normalize (16kHz Mono)      │
          │  2. Wav2Vec 2.0 CTC Forward Pass (PyTorch CUDA)  │
          │  3. Model Returned Cleanly to Host CPU           │
          └────────────────────────┬─────────────────────────┘
                                   │ CTC Logits Matrix
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │  In-Process CPU Host Pipeline                    │
          │  1. KenLM 3-Gram Beam Search (Native C++ ELF)    │
          │  2. Ephemeral In-Memory Store (Zero Disk Leak)   │
          │  3. Gemini 2.5 Flash API Study Asset Generation  │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
          [ Rendered Spotify Karaoke Sync Player & Flashcards (:200 OK) ]
```

1. **Route Filtering & Root Takeover**: In `app.py`, Gradio's default landing routes are evicted, and the authentic Flask WSGI application is mounted at root `/` via `a2wsgi.WSGIMiddleware`.
2. **Hidden ZeroGPU Runner**: A headless Gradio Blocks queue endpoint (`/transcribe_zerogpu`) is exposed with `@spaces.GPU(duration=30)`.
3. **Token Forwarding**: When a visitor uploads lecture media, Flask extracts the visitor's `X-IP-Token` header and invokes the local endpoint via `gradio_client.Client` loopback, granting the NVIDIA A10G GPU slice on demand.
4. **Graceful Fail-Safe**: If visitor GPU quotas are ever exceeded, the system automatically degrades to CPU execution (`device="cpu"`) without throwing server errors.

> 📖 **Read the Full Deep-Dive**: For complete post-mortem analyses of the 4 major deployment incidents (Docker platform block, ZeroGPU context blackout, KenLM native C++ compilation, and Linux daemon multiprocessing locks), see [`docs/SYSTEMS_ARCHITECTURE_AND_POST_MORTEM.md`](docs/SYSTEMS_ARCHITECTURE_AND_POST_MORTEM.md).

---

## ✨ Key Features

### 🎧 Speech-to-Text Transcription
* **Acoustic Neural Model**: Fine-tuned Wav2Vec 2.0 trained on mixed-accent academic lecture audio.
* **Dual Decoding Options**:
  * **N-gram LM Beam Search**: KenLM language model integration for domain-specific phonetic error correction.
  * **Greedy CTC Decoding**: Ultra-fast baseline decoding.
* **Long-Audio Chunking**: Sliding-window chunking supporting lectures of 30+ minutes without GPU out-of-memory errors.

### 💻 User Interface & Experience
* **Spotify-Style Lyrics Synchronization**: Word-level and sentence-level timestamps that highlight dynamically as the audio plays.
* **Waveform Audio Player**: Clickable audio scrub bar that jumps directly to corresponding lecture sentences.
* **Responsive Dark / Light Themes**: Modern CSS design system featuring deep teal accents and clean typography.
* **Dual-View Modes**: Toggle between timestamped timeline segments and a clean continuous reading document view.
* **One-Click Export**: Export complete transcripts to **DOCX**, **TXT**, and **JSON**.

### 🧠 AI Study Companion (Powered by Google Gemini 2.5)
* **Executive Summary**: Structured lecture overview with key takeaways and bullet points.
* **Study Quiz**: Auto-generated multiple-choice questions with answer keys and explanations.
* **Technical Glossary**: Definitions for academic terminology extracted directly from transcript context.
* **Active-Recall Flashcards**: Interactive digital flip-cards for spaced-repetition study.
* **Lecture Q&A Chat**: Grounded conversational assistant answering student questions exclusively using lecture content.

---

## 📁 Repository Structure

```text
├── app.py                         # Root ASGI entrypoint & ZeroGPU Gradio Queue runner
├── packages.txt                   # Debian system toolchains (gcc, cmake, libboost, ffmpeg)
├── requirements.txt               # Python dependencies (KenLM C++ upstream source)
├── docs/
│   ├── SYSTEMS_ARCHITECTURE_AND_POST_MORTEM.md # Exhaustive technical post-mortem report
│   └── deploy_huggingface_spaces.md            # Step-by-step cloud deployment guide
├── scripts/
│   ├── deploy_to_hf_space.py      # Automated clean staging & sync to Hugging Face
│   └── upload_model_to_hf.py      # Tool to upload model checkpoints to Model Hub
├── src/
│   ├── pipeline/                  # Core ASR engine
│   │   ├── model.py               # ASRModel: Wav2Vec 2.0 + KenLM in-process decoder
│   │   ├── preprocessing.py       # Audio normalization, resampling & linear padding
│   │   └── config.py              # Acoustic pipeline hyperparameters
│   ├── web/                       # Authentic Web Application
│   │   ├── app.py                 # Flask routes, media conversion & ZeroGPU bridge
│   │   ├── ai_service.py          # Google Gemini 2.5 Flash API study assistant
│   │   ├── transcript_store.py    # Ephemeral store (Cloud) vs. persistent JSON (Local)
│   │   ├── static/                # CSS themes, JavaScript karaoke sync & branding
│   │   └── templates/             # Jinja2 HTML templates
│   ├── evaluate_pipeline.py       # Model WER / CER evaluation suite
│   ├── normalize_text.py          # Text cleaning and phonetic normalization
│   └── train_main_stable.py       # Acoustic model training harness
└── README.md
```

---

## 💻 Local Quickstart

To run ClassScript locally on your own machine with the **full persistent Library database** enabled:

### 1. Prerequisites
* Python 3.10 or 3.11
* FFmpeg installed and available on system `PATH`
* NVIDIA GPU with CUDA (optional, but accelerates local transcription)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/siddiqrossli/Wav2Vec2-Transformer-ASR-Lecture-Transcription-System.git
cd Wav2Vec2-Transformer-ASR-Lecture-Transcription-System

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Gemini AI (Optional)
To enable the AI Study Suite locally:
1. Create a `src/web/.env` file (copied from `src/web/.env.example`):
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   LLM_MODEL=gemini-2.5-flash
   ```
   *(Get a free key from [Google AI Studio](https://aistudio.google.com/apikey)).*

### 4. Launch Application
```bash
python src/web/serve.py
```
Open your browser at **`http://localhost:5000`**. You will see the authentic ClassScript interface with the **Library** management system fully active and transcripts saved to local disk (`src/web/data/transcripts/`).

---

## 📊 Evaluation & Benchmarks

The fine-tuned Wav2Vec 2.0 acoustic model was evaluated on held-out lecture speech test sets:

| Decoding Method | Word Error Rate (WER) | Character Error Rate (CER) | Inference Speed (A10G) |
| :--- | :---: | :---: | :---: |
| **Greedy CTC Baseline** | ~17.4% | ~7.2% | ~0.08x Real-Time |
| **Wav2Vec 2.0 + KenLM 3-Gram** | **~11.8%** | **~4.9%** | **~0.12x Real-Time** |

*Integrating the 3-gram language model yields a **~32% relative reduction in Word Error Rate**, resolving technical jargon, academic vocabulary, and homophones.*

---

## 📄 License & Attribution

This project is developed for academic, educational, and portfolio purposes by **Siddiq Rossli** as a Final Year Project for **Universiti Teknologi MARA (UiTM)**.

Distributed under the **MIT License**. See `LICENSE` for details.
