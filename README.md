# Audio Transcription and Text Analysis Pipeline

Complete pipeline for Spanish audio transcription using OpenAI Whisper and advanced text analysis with spaCy. Features economic and Argentinian lexicon extraction plus co-occurrence network analysis.

## Dependencies

Install ffmpeg (required by Whisper for audio processing):

```bash
sudo apt update
sudo apt install ffmpeg
```

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Spanish language model for spaCy:
```bash
python -m spacy download es_core_news_sm
```

## Usage

### Transcription
```bash
python transcribe.py <audio_file.mp3>
```

Example:
```bash
python transcribe.py ./audio/S08E05.mp3
```

### Text Analysis
Process transcriptions to create glossaries and co-occurrence graphs:
```bash
python episode_process.py <transcription.jsonl>
```

Example:
```bash
python episode_process.py ./outputs/S08E05_20250917_001758.jsonl
```

This creates:
- `glossary/economy_glossary.json` and `.md` - Economic terms with frequencies
- `glossary/argentinian_lexicon.json` and `.md` - Argentinian expressions with frequencies
- `outputs/{filename}_graph.html` - Interactive network visualization (unique per file)
- `outputs/graph_metrics.json` - Graph analysis metrics

## Features

### Transcription (transcribe.py)
- **Spanish language support** - Optimized for Spanish audio content
- **Multiple outputs** - Text (.txt) and structured segments (.jsonl)
- **Timestamped files** - Organized by date in outputs/ folder
- **Progress tracking** - Real-time transcription progress

### Text Analysis (episode_process.py)
- **Economic glossary** - Extracts financial/economic terminology
- **Argentinian lexicon** - Identifies regional expressions and cultural terms
- **Co-occurrence analysis** - Network graph of term relationships
- **Multiple formats** - JSON for data, Markdown for readability
- **Unique visualizations** - Separate graph files per analysis session

## File Structure
```
├── transcribe.py              # Audio transcription script
├── episode_process.py         # Text analysis pipeline
├── requirements.txt           # Python dependencies
├── outputs/                   # Transcription outputs
│   ├── {filename}_{date}.txt     # Plain text transcriptions
│   ├── {filename}_{date}.jsonl   # Segmented transcriptions
│   ├── {filename}_graph.html     # Network visualizations
│   └── graph_metrics.json        # Analysis metrics
└── glossary/                  # Domain-specific glossaries
    ├── economy_glossary.json/.md
    └── argentinian_lexicon.json/.md
```

## Deactivate Virtual Environment

When done:
```bash
deactivate
```