# Audio Transcription and Text Analysis Pipeline

Complete pipeline for Spanish audio transcription using OpenAI Whisper and advanced text analysis with spaCy. Features modular economic term detection, Argentinian lexicon extraction, and co-occurrence network analysis with machine learning capabilities.

## Quick Start with Modern CLI

The fastest way to get started is with the unified `my-transcript` command:

```bash
# Install the package
pip install -e .

# Transcribe audio
my-transcript transcribe interview.mp3

# Analyze transcription
my-transcript analyze outputs/interview_20250917.jsonl

# Detect economic terms
my-transcript detect outputs/interview_20250917.jsonl

# View configuration
my-transcript config --show
```

## Dependencies

Install ffmpeg (required by Whisper for audio processing):

```bash
sudo apt update
sudo apt install ffmpeg
```

## Setup

### Option 1: Modern Package Installation (Recommended)

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install as package with all dependencies:
```bash
pip install -e .
```

3. Install Spanish language model for spaCy:
```bash
python -m spacy download es_core_news_sm
```

4. Verify installation:
```bash
my-transcript --help
```

### Option 2: Traditional Script Usage

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

### Modern CLI Interface (Recommended)

#### Transcription
```bash
my-transcript transcribe <audio_file>
```

**Examples:**
```bash
# Basic transcription
my-transcript transcribe interview.mp3

# Custom model and language
my-transcript transcribe --model small --language Spanish podcast.wav

# Verbose output
my-transcript --verbose transcribe interview.mp3
```

#### Text Analysis
```bash
my-transcript analyze <transcription.jsonl>
```

**Examples:**
```bash
# Basic analysis
my-transcript analyze outputs/interview_20250917.jsonl

# Custom parameters
my-transcript analyze --window-size 7 --freq-threshold 2 transcription.jsonl
```

#### Economic Term Detection
```bash
my-transcript detect <transcription.jsonl>
```

**Examples:**
```bash
# Basic detection
my-transcript detect outputs/interview_20250917.jsonl

# Custom similarity threshold
my-transcript detect --similarity-threshold 0.8 transcription.jsonl

# Disable embeddings
my-transcript detect --no-embeddings transcription.jsonl
```

#### Configuration Management
```bash
# Show current configuration
my-transcript config --show

# Validate configuration
my-transcript config --validate

# Get help for any command
my-transcript transcribe --help
```

### Legacy Script Usage (Backward Compatible)

All original scripts continue to work unchanged:

#### Transcription
```bash
python transcribe.py <audio_file.mp3>
```

Example:
```bash
python transcribe.py ./audio/S08E05.mp3
```

#### Text Analysis
```bash
python episode_process.py <transcription.jsonl>
```

Example:
```bash
python episode_process.py ./outputs/S08E05_20250917_001758.jsonl
```

#### Economic Term Detection
```bash
python detect_economic_terms_with_embeddings.py <transcription.jsonl>
```

**Output Files Created:**
- `glossary/economy_glossary.json` and `.md` - Economic terms with frequencies
- `glossary/argentinian_lexicon.json` and `.md` - Argentinian expressions with frequencies
- `outputs/{filename}_graph.html` - Interactive network visualization (unique per file)
- `outputs/{filename}_graph_metrics.json` - Graph analysis metrics
- `outputs/{filename}_detected_terms.json` and `.md` - ML-detected economic terms

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

## Project Structure

```
my-transcript/                 # Audio transcription pipeline (1.4MB total)
├── pyproject.toml            # Modern Python packaging configuration
├── requirements.txt          # Python dependencies (legacy support)
├── cli/                      # Modern CLI interface
│   ├── __init__.py
│   └── main.py              # Unified my-transcript command
├── transcribe.py             # Audio transcription script (legacy)
├── episode_process.py        # Text analysis pipeline (legacy)
├── detect_economic_terms_with_embeddings.py  # ML detection (legacy)
├── config/                   # Configuration management
│   ├── __init__.py
│   ├── config_loader.py     # Hierarchical configuration
│   └── settings.json        # Default settings
├── detectors/                # Term detection modules
│   ├── __init__.py
│   └── economic_detector.py # SBERT + FAISS detector
├── extractors/               # Data extraction modules
│   ├── __init__.py
│   └── numeric_extractor.py # Spanish numeric processing
├── models/                   # Data models and metrics
│   ├── __init__.py
│   ├── detected_term.py     # DetectedTerm dataclass
│   └── performance_metrics.py # Performance tracking
├── tools/                    # Development utilities
│   ├── README.md            # Tool documentation
│   ├── diagnostics/         # Diagnostic tools
│   │   └── tune_similarity_threshold.py
│   └── validation/          # Validation tools
│       └── validate_extraction.py
├── archive/                  # Completed utilities
│   └── backup_system.py
├── docs/                     # Project documentation
│   ├── cleanup.md           # Project cleanup analysis
│   └── technical_restructuring_plan.md  # Implementation plan
├── tests/                    # Comprehensive test suite
│   ├── unit/                # Component testing
│   ├── integration/         # Cross-component validation
│   └── e2e/                 # End-to-end scenarios
├── outputs/                  # Generated outputs (gitignored)
│   ├── {filename}_{date}.txt     # Plain text transcriptions
│   ├── {filename}_{date}.jsonl   # Segmented transcriptions
│   ├── {filename}_graph.html     # Network visualizations (CDN-based)
│   ├── {filename}_graph_metrics.json  # Analysis metrics
│   └── {filename}_detected_terms.json/.md  # ML detection results
└── glossary/                 # Domain-specific glossaries
    ├── economy_glossary.json/.md        # Economic terms
    └── argentinian_lexicon.json/.md     # Argentinian expressions
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
- CLI functionality equivalence

## Development Tools

The project includes specialized development utilities in the `tools/` directory:

```bash
# Diagnostic tools
python tools/diagnostics/tune_similarity_threshold.py

# Validation tools
python tools/validation/validate_extraction.py
```

See `tools/README.md` for detailed documentation.

## Troubleshooting

### Common Issues

**CLI Command Not Found:**
```bash
# Ensure package is installed in current environment
pip install -e .

# Verify installation
my-transcript --help
```

**spaCy Model Missing:**
```bash
# Install Spanish language model
python -m spacy download es_core_news_sm

# Verify installation
python -c "import spacy; spacy.load('es_core_news_sm')"
```

**Configuration Issues:**
```bash
# Validate configuration
my-transcript config --validate

# Show current settings
my-transcript config --show
```

**Memory Issues with Large Audio Files:**
- Use smaller Whisper models: `--model tiny` or `--model base`
- Process audio in smaller segments
- Ensure sufficient RAM available

**FAISS Import Errors:**
```bash
# Reinstall faiss-cpu
pip uninstall faiss-cpu
pip install faiss-cpu
```

### Performance Optimization

**Faster Transcription:**
- Use `--model tiny` for quick testing
- Use GPU-enabled Whisper for large files
- Consider audio preprocessing to reduce file size

**Faster Analysis:**
- Adjust `--freq-threshold` to filter less frequent terms
- Reduce `--window-size` for faster co-occurrence analysis
- Use `--no-embeddings` to disable ML similarity matching

**Network Visualizations:**
- Graph HTML files now use CDN resources for vis.js libraries
- No local JavaScript dependencies required
- Visualizations work offline after initial load

## Project Improvements

### Recent Enhancements
- ✅ **CLI Unification**: Modern `my-transcript` command with subcommands
- ✅ **Package Distribution**: PyPI-ready with `pip install -e .`
- ✅ **Code Organization**: Clean separation of tools, docs, and core modules
- ✅ **Size Optimization**: 756KB reduction by using CDN resources for visualizations
- ✅ **Documentation**: Comprehensive docs in dedicated `docs/` directory
- ✅ **Zero Breaking Changes**: All legacy scripts remain functional

### Architecture Highlights
- **Modular Design**: Clean separation between config, detectors, extractors, and models
- **Backward Compatibility**: Legacy scripts preserved alongside modern CLI
- **Professional Packaging**: Standard Python project structure with pyproject.toml
- **Development Tools**: Organized diagnostic and validation utilities
- **Comprehensive Testing**: Unit, integration, and end-to-end test coverage

## Migration from Legacy Scripts

The modern CLI provides the same functionality as legacy scripts:

| Legacy Script | Modern CLI Equivalent |
|---------------|----------------------|
| `python transcribe.py file.mp3` | `my-transcript transcribe file.mp3` |
| `python episode_process.py file.jsonl` | `my-transcript analyze file.jsonl` |
| `python detect_economic_terms_with_embeddings.py file.jsonl` | `my-transcript detect file.jsonl` |

All legacy scripts continue to work unchanged for backward compatibility.

## Deactivate Virtual Environment

When done:
```bash
deactivate
```