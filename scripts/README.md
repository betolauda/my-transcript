# Legacy Scripts

This directory contains the original standalone scripts that provide backward compatibility while the modern CLI interface is available via the `my-transcript` command.

## Available Scripts

### `transcribe.py` - Audio Transcription
Converts audio files to text using OpenAI Whisper with Spanish language optimization.

**Usage:**
```bash
python scripts/transcribe.py <audio_file>
```

**Example:**
```bash
python scripts/transcribe.py interview.mp3
```

**Outputs:**
- `outputs/{filename}_{date}.txt` - Plain text transcription
- `outputs/{filename}_{date}.jsonl` - Segmented transcription with timestamps

### `episode_process.py` - NLP Analysis Pipeline
Processes transcriptions to extract economic and cultural terms, creates glossaries and co-occurrence network graphs.

**Usage:**
```bash
python scripts/episode_process.py <transcription.jsonl>
```

**Example:**
```bash
python scripts/episode_process.py outputs/interview_20250917.jsonl
```

**Outputs:**
- `glossary/economy_glossary.json/.md` - Economic terms with frequencies
- `glossary/argentinian_lexicon.json/.md` - Argentinian expressions
- `outputs/{filename}_graph.html` - Interactive network visualization
- `outputs/{filename}_graph_metrics.json` - Graph analysis metrics

### `detect_economic_terms_with_embeddings.py` - ML Term Detection
Advanced economic term detection using SBERT embeddings and FAISS similarity search.

**Usage:**
```bash
python scripts/detect_economic_terms_with_embeddings.py <transcription.jsonl>
```

**Example:**
```bash
python scripts/detect_economic_terms_with_embeddings.py outputs/interview_20250917.jsonl
```

**Outputs:**
- `outputs/{filename}_detected_terms.json` - ML-detected terms (JSON)
- `outputs/{filename}_detected_terms.md` - ML-detected terms (Markdown)

## Modern CLI Alternative

These scripts are preserved for backward compatibility. For new workflows, consider using the modern CLI interface:

```bash
# Modern CLI equivalents
my-transcript transcribe interview.mp3
my-transcript analyze outputs/interview_20250917.jsonl
my-transcript detect outputs/interview_20250917.jsonl
```

## Dependencies

All scripts require the same dependencies as specified in `requirements.txt`:
- openai-whisper
- spacy (with es_core_news_sm model)
- networkx, pyvis
- sentence-transformers, faiss-cpu
- regex

## Migration Notes

- All scripts maintain identical functionality to their original versions
- File paths and output formats remain unchanged
- Configuration system works identically across CLI and scripts
- Scripts can be run from any directory (use relative paths to scripts/)