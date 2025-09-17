#!/usr/bin/env python3
"""
Configuration Loader for Economic Term Detection

Loads and validates JSON configuration with fallback defaults.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration for economic term detection."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.

        Args:
            config_path: Path to JSON config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/settings.json relative to this file
            config_path = Path(__file__).parent / "settings.json"

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file with error handling."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self._config = self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}. Using defaults.")
            self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration as fallback."""
        return {
            "models": {
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "spacy_models": ["es_core_news_trf", "es_core_news_md", "es_core_news_sm"]
            },
            "detection": {
                "use_embeddings": True,
                "similarity_threshold": 0.75,
                "distance_threshold": 10,
                "top_k": 3,
                "context_window": 20
            },
            "output_dirs": {
                "glossary": "glossary",
                "analysis": "outputs"
            },
            "performance_tolerances": {
                "max_degradation_percent": 5.0,
                "memory_tolerance_mb": 50
            },
            "canonical_terms": {}
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'models.embedding_model')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._config is None:
            return default

        # Handle dot notation for nested keys
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_models_config(self) -> Dict[str, Any]:
        """Get models configuration section."""
        return self.get('models', {})

    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration section."""
        return self.get('detection', {})

    def get_canonical_terms(self) -> Dict[str, Any]:
        """Get canonical terms dictionary."""
        return self.get('canonical_terms', {})

    def get_output_dirs(self) -> Dict[str, str]:
        """Get output directories configuration."""
        return self.get('output_dirs', {"glossary": "glossary", "analysis": "outputs"})

    def get_performance_tolerances(self) -> Dict[str, float]:
        """Get performance tolerance settings."""
        return self.get('performance_tolerances', {"max_degradation_percent": 5.0, "memory_tolerance_mb": 50})

    # Convenience methods for frequently used values
    @property
    def use_embeddings(self) -> bool:
        """Whether to use embeddings for semantic matching."""
        return self.get('detection.use_embeddings', True)

    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        return self.get('models.embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')

    @property
    def spacy_models(self) -> list:
        """List of spaCy models to try."""
        return self.get('models.spacy_models', ['es_core_news_sm'])

    @property
    def similarity_threshold(self) -> float:
        """Similarity threshold for semantic matching."""
        return self.get('detection.similarity_threshold', 0.75)

    @property
    def distance_threshold(self) -> int:
        """Distance threshold for numeric association."""
        return self.get('detection.distance_threshold', 10)

    @property
    def top_k(self) -> int:
        """Number of top candidates to consider."""
        return self.get('detection.top_k', 3)

    @property
    def context_window(self) -> int:
        """Context window size in characters."""
        return self.get('detection.context_window', 20)

    def reload(self):
        """Reload configuration from file."""
        self._load_config()

    def validate(self) -> bool:
        """Validate configuration structure and values.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self._config is None:
            return False

        required_sections = ['models', 'detection', 'output_dirs', 'canonical_terms']
        for section in required_sections:
            if section not in self._config:
                logger.error(f"Missing required configuration section: {section}")
                return False

        # Validate detection settings
        detection = self._config.get('detection', {})
        if not isinstance(detection.get('similarity_threshold'), (int, float)):
            logger.error("Invalid similarity_threshold: must be numeric")
            return False

        if not (0.0 <= detection.get('similarity_threshold', 0) <= 1.0):
            logger.error("Invalid similarity_threshold: must be between 0.0 and 1.0")
            return False

        return True


# Global configuration instance
_global_config = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Get global configuration instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    return _global_config


def reload_config():
    """Reload global configuration from file."""
    global _global_config
    if _global_config is not None:
        _global_config.reload()