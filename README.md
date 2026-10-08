# ASR Lecture Transcriber & Study Assistant

An end-to-end Automatic Speech Recognition (ASR) system designed for academic lecture transcription, featuring acoustic modeling, n-gram language model decoding, and an AI-powered study companion.

## Features

- **Acoustic Modeling**: Fine-tuned Wav2Vec 2.0 architecture optimized for lecture speech.
- **Dual Decoding Pipeline**:
  - **Greedy CTC Decoding**: Fast, lightweight baseline decoding.
  - **N-gram LM Beam Search**: Enhanced transcription accuracy using language model rescoring.
- **Web Interface (ClassScript / ASR Studio)**:
  - Interactive audio player synchronized with timestamped segments.
  - Real-time search across lecture segments.
  - Dual layout view: Timestamped segments vs. continuous paragraph reading mode.
  - One-click export to **DOCX**, **TXT**, and **JSON**.
  - Lecture Library manager with bulk selection and deletion.
- **AI Study Assistant (Google Gemini)**:
  - **AI Summary**: Structured lecture overview with key points and takeaways.
  - **Study Quiz**: Auto-generated multiple-choice questions with answer keys and explanations.
  - **Glossary**: Extracted technical definitions grounded in lecture content.
  - **Flashcards**: Interactive study cards for spaced repetition.
  - **Lecture Q&A Chat**: Grounded Q&A answering questions exclusively from the transcript.

---

## Project Structure

```text
├── data/                      # Input audio & output directories
├── src/
│   ├── pipeline/              # Audio preprocessing, model definitions & decoding
│   ├── web/                   # Flask web interface & study assistant
│   │   ├── static/            # CSS styles, JavaScript assets & images
│   │   ├── templates/         # Jinja2 HTML templates
│   │   ├── ai_service.py      # Gemini AI integration layer
│   │   ├── app.py             # Flask application entry point
│   │   └── .env.example       # Environment template for AI configuration
│   ├── evaluate_pipeline.py   # WER/CER evaluation scripts
│   ├── normalize_text.py      # Text cleaning and normalization utilities
│   └── train_main_stable.py   # Acoustic model training scripts
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10 or 3.11 recommended
- CUDA-enabled GPU (optional, but recommended for faster transcription)
- FFmpeg (for audio format conversions)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd asr-lecture-transcriber

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Gemini AI (Optional)

To enable the AI Study Assistant (Summary, Quiz, Glossary, Flashcards, Chat):

1. Copy the example environment file:
   ```bash
   cp src/web/.env.example src/web/.env
   ```
2. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. Set your key in `src/web/.env`:
   ```env
   GEMINI_API_KEY=your-gemini-api-key-here
   LLM_MODEL=gemini-3.5-flash
   ```

### 4. Running the Web Application

```bash
python src/web/app.py
```

Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## License

This project is developed for academic and educational purposes.
