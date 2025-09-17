# Project Structure Cleanup Plan

## Current State Analysis

After analyzing the project structure, I've identified several areas needing cleanup following the recent comprehensive refactoring:

### Current Issues
1. **Massive backup system overhead** (1.1MB across 5 phase backups with complete duplicates)
2. **Scattered test files** - 10 test files, some duplicated between root and tests/ directory
3. **Root directory clutter** - Many standalone test/validation scripts
4. **Missing package structure** - No setup.py, proper __init__.py files missing
5. **Mixed naming conventions** - Inconsistent file/directory naming
6. **Outdated documentation** - README doesn't reflect new modular architecture

## Target Structure

```
my-transcript/
├── src/
│   └── my_transcript/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── transcribe.py
│       │   └── detect_terms.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── config_loader.py
│       │   └── settings.json
│       ├── detectors/
│       ├── extractors/
│       ├── models/
│       └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
├── setup.py
├── pyproject.toml
├── README.md
├── requirements.txt
└── examples/
```

## 3-Phase Cleanup Plan

### Phase 1: Backup System Cleanup (CRITICAL - 80% storage reduction)
**Priority**: CRITICAL
**Impact**: High storage savings, no functionality loss
**Risk**: Low (backed up in git)

1. **Archive strategy**:
   - Create single compressed archive: `refactor_history.tar.gz`
   - Keep only phase5 backup (final working state before extraction)
   - Remove 4 intermediate phase backups

2. **Validation files consolidation**:
   - Keep `test_functional_equivalence.py` as canonical validation
   - Remove `test_before_after_comparison.py` (redundant)
   - Remove standalone `test_economic_detector.py` (superseded by tests/unit/)

### Phase 2: Test Organization (HIGH priority)
**Priority**: HIGH
**Impact**: Improved maintainability and clarity
**Risk**: Low (move operations with validation)

1. **Consolidate test structure**:
   - Move root-level test files to appropriate tests/ subdirectories
   - Standardize test naming: `test_<component>.py`
   - Remove duplicate test files

2. **Test categories**:
   - `tests/unit/` - Component-specific tests
   - `tests/integration/` - Cross-component tests
   - `tests/e2e/` - End-to-end validation
   - `tests/fixtures/` - Test data (move test_sample_segments.jsonl)

### Phase 3: Package Structure (MEDIUM priority)
**Priority**: MEDIUM
**Impact**: Professional Python package structure
**Risk**: Medium (requires import path updates)

1. **Create proper package structure**:
   - Create `src/my_transcript/` package structure
   - Add proper `__init__.py` files with version info
   - Move CLI scripts to `cli/` subpackage
   - Create `setup.py` and `pyproject.toml`

2. **Documentation improvements**:
   - Update README.md to reflect new modular architecture
   - Add proper docstrings and type hints
   - Create `docs/` directory with API documentation

## Implementation Timeline

- **Week 1**: Phase 1 (Backup cleanup) - Immediate 80% storage reduction
- **Week 2**: Phase 2 (Test organization) - Essential for maintainability
- **Week 3**: Phase 3 (Package structure) - Professional structure

## Success Criteria

1. **Storage reduction**: >75% reduction in repository size
2. **Test clarity**: All tests properly categorized and no duplicates
3. **Import compatibility**: All existing functionality preserved
4. **Package structure**: Installable via pip with proper entry points
5. **Documentation**: Updated to reflect new architecture

## Risk Mitigation

- Git history preservation for all changes
- Comprehensive test suite execution after each phase
- Incremental implementation with validation checkpoints
- Rollback procedures documented for each phase

## Detailed Implementation Steps

### Phase 1 Detailed Steps

#### 1.1 Create Backup Archive
```bash
# Create compressed archive of all backup phases
tar -czf refactor_history.tar.gz refactor_backups/
```

#### 1.2 Backup Cleanup
```bash
# Keep only the final phase backup for emergency rollback
mv refactor_backups/phase5_extract_economicdetector_20250917_104453 refactor_final_backup
rm -rf refactor_backups/
mkdir refactor_backups
mv refactor_final_backup refactor_backups/phase5_final
```

#### 1.3 Remove Redundant Test Files
```bash
# Remove redundant validation files
rm test_before_after_comparison.py
rm test_economic_detector.py  # Superseded by tests/unit/test_economic_detector.py
```

### Phase 2 Detailed Steps

#### 2.1 Test Structure Reorganization
```bash
# Create proper test structure
mkdir -p tests/fixtures tests/integration

# Move test data to fixtures
mv test_sample_segments.jsonl tests/fixtures/

# Move functional equivalence test
mv test_functional_equivalence.py tests/integration/

# Verify no duplicate test files remain in root
```

#### 2.2 Test File Consolidation
- Ensure all unit tests are in `tests/unit/`
- Ensure all integration tests are in `tests/integration/`
- Ensure all e2e tests are in `tests/e2e/`
- Remove any remaining test files from root directory

### Phase 3 Detailed Steps

#### 3.1 Package Structure Creation
```bash
# Create src layout
mkdir -p src/my_transcript/cli src/my_transcript/utils

# Move main modules to package
mv config/ detectors/ extractors/ models/ src/my_transcript/

# Move CLI scripts
mv transcribe.py src/my_transcript/cli/
mv detect_economic_terms_with_embeddings.py src/my_transcript/cli/detect_terms.py
mv episode_process.py src/my_transcript/cli/
```

#### 3.2 Package Configuration
Create `setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="my-transcript",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "openai-whisper",
        "spacy",
        "numpy",
        "torch",
        "sentence-transformers",
        "faiss-cpu",
        "networkx",
        "pyvis"
    ],
    entry_points={
        "console_scripts": [
            "transcribe=my_transcript.cli.transcribe:main",
            "detect-terms=my_transcript.cli.detect_terms:main",
            "process-episode=my_transcript.cli.episode_process:main",
        ],
    },
)
```

Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-transcript"
version = "1.0.0"
description = "Audio transcription and economic term detection pipeline"
readme = "README.md"
requires-python = ">=3.8"
```

#### 3.3 Documentation Updates
- Update README.md to reflect new package structure
- Create docs/ directory with proper API documentation
- Add type hints and improved docstrings throughout codebase

## File Mapping

### Current → Target Structure

```
Current Structure                    →  Target Structure
=====================================   =====================================
./transcribe.py                     →  src/my_transcript/cli/transcribe.py
./episode_process.py                →  src/my_transcript/cli/episode_process.py
./detect_economic_terms_with_*.py   →  src/my_transcript/cli/detect_terms.py
./config/                           →  src/my_transcript/config/
./detectors/                        →  src/my_transcript/detectors/
./extractors/                       →  src/my_transcript/extractors/
./models/                           →  src/my_transcript/models/
./tests/                            →  tests/ (reorganized)
./test_functional_equivalence.py    →  tests/integration/test_functional_equivalence.py
./test_sample_segments.jsonl        →  tests/fixtures/test_sample_segments.jsonl
./refactor_backups/                 →  refactor_history.tar.gz + minimal backup
```

## Validation Checklist

After each phase, verify:

### Phase 1 Validation
- [ ] Repository size reduced by >75%
- [ ] Git history intact
- [ ] All functionality tests pass
- [ ] Single emergency backup retained

### Phase 2 Validation
- [ ] All tests properly categorized
- [ ] No duplicate test files
- [ ] Test suite runs successfully
- [ ] Clear test organization structure

### Phase 3 Validation
- [ ] Package installs correctly (`pip install -e .`)
- [ ] All CLI commands work via entry points
- [ ] Import paths updated and functional
- [ ] Documentation reflects new structure

## Emergency Rollback Procedures

### Phase 1 Rollback
```bash
# Restore from archive if needed
tar -xzf refactor_history.tar.gz
# Restore git state if needed
git reset --hard <previous-commit>
```

### Phase 2 Rollback
```bash
# Move files back to original locations
# All moves are reversible
```

### Phase 3 Rollback
```bash
# Move packages back to root
mv src/my_transcript/* ./
rmdir src/my_transcript src/
# Remove package files
rm setup.py pyproject.toml
```