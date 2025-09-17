#!/usr/bin/env python3
"""
Unit tests for EconomicTermDetector

Tests the extracted EconomicTermDetector class functionality.
"""

import unittest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_loader import ConfigLoader


class TestEconomicDetector(unittest.TestCase):
    """Test EconomicTermDetector class functionality."""

    def setUp(self):
        """Set up test fixtures with mock configuration."""
        # Create a test configuration
        self.test_config_data = {
            "models": {
                "embedding_model": "test-model",
                "spacy_models": ["test_model"]
            },
            "detection": {
                "use_embeddings": False,  # Disable to avoid dependencies
                "similarity_threshold": 0.8,
                "distance_threshold": 15,
                "top_k": 2,
                "context_window": 30
            },
            "output_dirs": {
                "glossary": "test_glossary",
                "analysis": "test_outputs"
            },
            "canonical_terms": {
                "test_term": {
                    "labels": ["test", "prueba"],
                    "description": "Test term"
                },
                "inflacion": {
                    "labels": ["inflación", "inflation"],
                    "description": "Inflation rate"
                }
            }
        }

        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_config_data, self.temp_config)
        self.temp_config.close()

        # Create config loader with test data
        self.config = ConfigLoader(self.temp_config.name)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary config file
        Path(self.temp_config.name).unlink()

    def test_config_loader_basic_functionality(self):
        """Test that ConfigLoader works correctly."""
        self.assertEqual(self.config.use_embeddings, False)
        self.assertEqual(self.config.similarity_threshold, 0.8)
        self.assertEqual(self.config.distance_threshold, 15)
        self.assertEqual(self.config.context_window, 30)

        canonical_terms = self.config.get_canonical_terms()
        self.assertIn("test_term", canonical_terms)
        self.assertIn("inflacion", canonical_terms)

    def test_config_validation(self):
        """Test configuration validation."""
        self.assertTrue(self.config.validate())

        # Test invalid config
        invalid_config_data = {
            "detection": {
                "similarity_threshold": 1.5  # Invalid: > 1.0
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config_data, f)
            f.flush()
            invalid_config = ConfigLoader(f.name)
            self.assertFalse(invalid_config.validate())
            Path(f.name).unlink()

    @patch('detectors.economic_detector.spacy')
    def test_economic_detector_initialization_no_spacy(self, mock_spacy):
        """Test EconomicTermDetector initialization when spaCy is not available."""
        # Test that import works even when spaCy is not available
        mock_spacy.load.side_effect = OSError("spaCy model not found")
        mock_spacy.blank.return_value = Mock()

        try:
            from detectors.economic_detector import EconomicTermDetector
            detector = EconomicTermDetector(self.config)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.use_embeddings, False)
            self.assertIsNone(detector.sentence_model)
        except ImportError:
            # Skip test if spaCy is not available at all
            self.skipTest("spaCy not available for testing")

    def test_config_dot_notation_access(self):
        """Test configuration access with dot notation."""
        self.assertEqual(self.config.get('models.embedding_model'), 'test-model')
        self.assertEqual(self.config.get('detection.similarity_threshold'), 0.8)
        self.assertEqual(self.config.get('nonexistent.key', 'default'), 'default')

    def test_config_convenience_properties(self):
        """Test configuration convenience properties."""
        self.assertEqual(self.config.embedding_model, 'test-model')
        self.assertEqual(self.config.spacy_models, ['test_model'])
        self.assertEqual(self.config.similarity_threshold, 0.8)
        self.assertEqual(self.config.distance_threshold, 15)
        self.assertEqual(self.config.top_k, 2)
        self.assertEqual(self.config.context_window, 30)

    def test_config_default_fallbacks(self):
        """Test configuration default fallbacks."""
        # Test with empty config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            f.flush()
            empty_config = ConfigLoader(f.name)

            # Should use defaults
            self.assertEqual(empty_config.use_embeddings, True)
            self.assertEqual(empty_config.similarity_threshold, 0.75)
            self.assertEqual(empty_config.get_canonical_terms(), {})

            Path(f.name).unlink()

    def test_config_sections_access(self):
        """Test configuration section access methods."""
        models_config = self.config.get_models_config()
        self.assertIn('embedding_model', models_config)
        self.assertEqual(models_config['embedding_model'], 'test-model')

        detection_config = self.config.get_detection_config()
        self.assertIn('similarity_threshold', detection_config)
        self.assertEqual(detection_config['similarity_threshold'], 0.8)

        output_dirs = self.config.get_output_dirs()
        self.assertIn('analysis', output_dirs)
        self.assertEqual(output_dirs['analysis'], 'test_outputs')

    @patch('detectors.economic_detector.EMBEDDINGS_AVAILABLE', False)
    @patch('detectors.economic_detector.spacy')
    def test_detector_without_embeddings(self, mock_spacy):
        """Test EconomicTermDetector functionality without embeddings."""
        # Mock spaCy components
        mock_nlp = Mock()
        mock_nlp.vocab = Mock()
        mock_nlp.pipe_names = []
        mock_spacy.load.side_effect = OSError("Model not found")
        mock_spacy.blank.return_value = mock_nlp

        try:
            from detectors.economic_detector import EconomicTermDetector

            # Disable embeddings in config
            config_no_embeddings = self.config
            config_no_embeddings._config['detection']['use_embeddings'] = False

            detector = EconomicTermDetector(config_no_embeddings)

            # Test initialization
            self.assertIsNotNone(detector)
            self.assertFalse(detector.use_embeddings)
            self.assertIsNone(detector.sentence_model)
            self.assertIsNone(detector.faiss_index)

        except ImportError:
            # Skip test if dependencies are not available
            self.skipTest("Required dependencies not available for testing")

    def test_context_extraction_logic(self):
        """Test context extraction logic separately."""
        # Test the context extraction method logic
        test_text = "This is a long text with economic terms like inflation rate in the middle."
        start_pos = 35  # Position of "economic"
        end_pos = 48    # Position of "terms"
        context_window = 10

        # Simulate context extraction
        context_start = max(0, start_pos - context_window)
        context_end = min(len(test_text), end_pos + context_window)
        context = test_text[context_start:context_end].strip()

        # Add ellipsis if context was truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(test_text):
            context = context + "..."

        self.assertIn("economic terms", context)
        self.assertTrue(context.startswith("..."))
        self.assertTrue(context.endswith("..."))

    def test_jsonl_processing_structure(self):
        """Test JSONL processing structure with mock data."""
        # Create test JSONL data
        test_data = [
            {"text": "La inflación subió 5%", "start": 0.0},
            {"text": "El PIB creció 2.5%", "start": 5.0},
            {"text": "", "start": 10.0},  # Empty text should be skipped
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in test_data:
                json.dump(record, f)
                f.write('\n')
            f.flush()

            # Test file exists and is readable
            self.assertTrue(Path(f.name).exists())

            # Read back and verify structure
            with open(f.name, 'r') as read_f:
                lines = read_f.readlines()
                self.assertEqual(len(lines), 3)

                parsed_records = []
                for line in lines:
                    if line.strip():
                        record = json.loads(line)
                        if record.get('text'):  # Skip empty text
                            parsed_records.append(record)

                self.assertEqual(len(parsed_records), 2)
                self.assertIn("inflación", parsed_records[0]['text'])
                self.assertIn("PIB", parsed_records[1]['text'])

            Path(f.name).unlink()

    def test_performance_metrics_integration(self):
        """Test that performance metrics are properly structured."""
        from models.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        # Test metric accumulation
        metrics.total_processing_time = 5.0
        metrics.total_segments = 100
        metrics.exact_matches = 25
        metrics.semantic_matches = 15
        metrics.numeric_associations = 10
        metrics.total_terms_detected = 50

        # Test calculations
        self.assertEqual(metrics.segments_per_second(), 20.0)
        self.assertEqual(metrics.terms_per_second(), 10.0)

        # Test that metrics can be serialized
        from dataclasses import asdict
        metrics_dict = asdict(metrics)
        self.assertIn('total_processing_time', metrics_dict)
        self.assertEqual(metrics_dict['exact_matches'], 25)

    def test_numeric_extractor_integration(self):
        """Test NumericExtractor integration with configuration."""
        from extractors.numeric_extractor import NumericExtractor

        # Test with custom distance threshold from config
        extractor = NumericExtractor(self.config.distance_threshold)

        test_text = "La tasa de inflación es del 8.5%"
        results = extractor.extract_numeric_values(test_text)

        self.assertTrue(len(results) > 0)
        # Should find the percentage
        percentage_found = any(r['type'] == 'percentage' for r in results)
        self.assertTrue(percentage_found)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)