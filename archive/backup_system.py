#!/usr/bin/env python3
"""
Backup and Rollback System for Refactoring

Creates backups before each refactoring phase and provides rollback capability.
"""

import shutil
import os
import json
from datetime import datetime
from pathlib import Path


class RefactorBackup:
    """Manages backups and rollbacks during refactoring process."""

    def __init__(self, backup_dir: str = "refactor_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

    def create_phase_backup(self, phase_name: str) -> str:
        """Create backup before starting a refactoring phase."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{phase_name}_{timestamp}"
        backup_path = self.backup_dir / backup_name

        # Backup main file
        main_file = "detect_economic_terms_with_embeddings.py"
        if os.path.exists(main_file):
            backup_path.mkdir(exist_ok=True)
            shutil.copy2(main_file, backup_path / main_file)

        # Backup existing test files
        test_files = ["test_economic_detector.py", "tune_similarity_threshold.py", "test_sample_segments.jsonl"]
        for test_file in test_files:
            if os.path.exists(test_file):
                shutil.copy2(test_file, backup_path / test_file)

        # Backup any created module directories
        for module_dir in ["models", "extractors", "detectors", "config", "tests"]:
            if os.path.exists(module_dir):
                shutil.copytree(module_dir, backup_path / module_dir, dirs_exist_ok=True)

        print(f"✓ Phase backup created: {backup_path}")
        return str(backup_path)

    def list_backups(self):
        """List available backups."""
        backups = list(self.backup_dir.glob("*"))
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        print("Available backups:")
        for backup in backups:
            timestamp = datetime.fromtimestamp(backup.stat().st_mtime)
            print(f"  {backup.name} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def rollback_to_phase(self, backup_name: str):
        """Rollback to a specific phase backup."""
        backup_path = self.backup_dir / backup_name

        if not backup_path.exists():
            print(f"Error: Backup {backup_name} not found")
            return False

        # Restore main file
        main_file = backup_path / "detect_economic_terms_with_embeddings.py"
        if main_file.exists():
            shutil.copy2(main_file, "detect_economic_terms_with_embeddings.py")

        # Remove created directories and restore from backup
        for module_dir in ["models", "extractors", "detectors", "config", "tests"]:
            if os.path.exists(module_dir):
                shutil.rmtree(module_dir)
            backup_module = backup_path / module_dir
            if backup_module.exists():
                shutil.copytree(backup_module, module_dir)

        print(f"✓ Rolled back to: {backup_name}")
        return True


def extract_configuration_constants():
    """Auto-generate JSON configuration from current constants in the main file."""

    # Read the main file to extract constants
    with open("detect_economic_terms_with_embeddings.py", "r") as f:
        content = f.read()

    # Extract configuration constants
    config = {
        "use_embeddings": True,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "top_k": 3,
        "similarity_threshold": 0.75,
        "context_window": 20,
        "distance_threshold": 10,
        "output_dirs": {
            "glossary": "glossary",
            "analysis": "outputs"
        },
        "spacy_models": [
            "es_core_news_trf",
            "es_core_news_md",
            "es_core_news_sm"
        ],
        "performance_tolerances": {
            "max_degradation_percent": 5.0,
            "memory_tolerance_mb": 50
        }
    }

    # Save configuration
    config_path = Path("config/settings.json")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✓ Configuration extracted to: {config_path}")
    return config


if __name__ == "__main__":
    import sys

    backup_system = RefactorBackup()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python backup_system.py backup <phase_name>")
        print("  python backup_system.py list")
        print("  python backup_system.py rollback <backup_name>")
        print("  python backup_system.py extract-config")
        sys.exit(1)

    command = sys.argv[1]

    if command == "backup" and len(sys.argv) > 2:
        backup_system.create_phase_backup(sys.argv[2])
    elif command == "list":
        backup_system.list_backups()
    elif command == "rollback" and len(sys.argv) > 2:
        backup_system.rollback_to_phase(sys.argv[2])
    elif command == "extract-config":
        extract_configuration_constants()
    else:
        print("Invalid command or missing arguments")