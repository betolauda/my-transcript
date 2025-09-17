# Development Tools

This directory contains utility tools for development, debugging, and validation of the my-transcript pipeline.

## Diagnostics (`tools/diagnostics/`)

### `tune_similarity_threshold.py` - ML Similarity Threshold Optimization
Advanced utility for optimizing SBERT similarity thresholds in economic term detection.

**Usage:**
```bash
python tools/diagnostics/tune_similarity_threshold.py
```

**Purpose:**
- Analyze similarity distributions across economic term corpus
- Find optimal threshold values for precision/recall balance
- Generate performance metrics and visualizations
- Validate ML model configuration

**Requirements:**
- Existing JSONL transcription files
- Pre-trained SBERT embeddings model
- Configuration system setup

## Validation (`tools/validation/`)

### `validate_extraction.py` - Import and Extraction Validation
Utility for validating import paths and data extraction functionality.

**Usage:**
```bash
python tools/validation/validate_extraction.py
```

**Purpose:**
- Verify module import paths are correct
- Validate data extraction pipeline integrity
- Test configuration loading and validation
- Ensure model initialization works properly

**Use Cases:**
- Pre-deployment validation
- CI/CD pipeline integration
- Development environment verification
- Troubleshooting import issues

## Archive (`archive/`)

Contains historical utilities that have completed their purpose and are no longer needed for current development:

- `backup_system.py` - Refactoring backup utility (completed Phase 1 cleanup)

## Usage Guidelines

### Running Diagnostic Tools
```bash
# Activate virtual environment
source venv/bin/activate

# Run similarity threshold tuning
python tools/diagnostics/tune_similarity_threshold.py

# Validate extraction functionality
python tools/validation/validate_extraction.py
```

### Integration with Main Pipeline
These tools are designed to work alongside the main pipeline components:

- **Compatible with CLI**: Tools can be run independently of the new `my-transcript` CLI
- **Uses same configuration**: Tools respect the existing `config/` system
- **Output coordination**: Tools save results to appropriate output directories

### Development Workflow
1. **Pre-development**: Run validation tools to ensure environment setup
2. **During development**: Use diagnostic tools to optimize parameters
3. **Post-development**: Validate changes don't break existing functionality

## Future Tools

Additional diagnostic and validation tools can be added to this structure:

- `tools/benchmarks/` - Performance benchmarking utilities
- `tools/testing/` - Specialized testing utilities
- `tools/deployment/` - Deployment validation tools
- `tools/analysis/` - Advanced analysis and reporting tools

## Notes

- All tools maintain the same Python version compatibility as the main pipeline (>=3.8)
- Tools use the existing virtual environment and dependencies
- Configuration changes in `config/` automatically affect tool behavior
- Tools follow the same coding standards and patterns as the main codebase