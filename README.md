# Audio Transcription and Text Analysis Pipeline

Complete pipeline for Spanish audio transcription using OpenAI Whisper and advanced text analysis with spaCy. Features modular economic term detection, Argentinian lexicon extraction, and co-occurrence network analysis with machine learning capabilities.

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
- **Modular architecture** - Configurable detectors and extractors
- **Economic term detection** - ML-powered financial/economic terminology extraction
- **Argentinian lexicon** - Identifies regional expressions and cultural terms
- **Co-occurrence analysis** - Network graph of term relationships
- **Machine learning** - Sentence transformers and FAISS for semantic similarity
- **Multiple formats** - JSON for data, Markdown for readability
- **Unique visualizations** - Separate graph files per analysis session

## File Structure
```
├── transcribe.py              # Audio transcription script
├── episode_process.py         # Text analysis pipeline
├── requirements.txt           # Python dependencies
├── config/                    # Configuration management
│   ├── __init__.py
│   └── config_loader.py
├── detectors/                 # Term detection modules
│   ├── __init__.py
│   └── economic_detector.py
├── extractors/                # Data extraction modules
│   ├── __init__.py
│   └── numeric_extractor.py
├── models/                    # Data models and metrics
│   ├── __init__.py
│   ├── detected_term.py
│   └── performance_metrics.py
├── tests/                     # Test suite
├── outputs/                   # Transcription outputs
│   ├── {filename}_{date}.txt     # Plain text transcriptions
│   ├── {filename}_{date}.jsonl   # Segmented transcriptions
│   ├── {filename}_graph.html     # Network visualizations
│   └── graph_metrics.json        # Analysis metrics
└── glossary/                  # Domain-specific glossaries
    ├── economy_glossary.json/.md
    └── argentinian_lexicon.json/.md
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

The project includes comprehensive tests for:
- Term detection accuracy
- Economic and cultural lexicon extraction
- Configuration loading
- Performance metrics validation

## Deactivate Virtual Environment

When done:
```bash
deactivate
```