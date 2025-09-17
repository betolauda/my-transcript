#!/usr/bin/env python3
"""
Unit tests for PerformanceMetrics model

Tests the extracted PerformanceMetrics dataclass functionality.
"""

import unittest
import sys
from pathlib import Path
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics dataclass functionality."""

    def test_performance_metrics_creation(self):
        """Test basic PerformanceMetrics creation with defaults."""
        metrics = PerformanceMetrics()

        # Test default values
        self.assertEqual(metrics.total_processing_time, 0.0)
        self.assertEqual(metrics.spacy_processing_time, 0.0)
        self.assertEqual(metrics.exact_matching_time, 0.0)
        self.assertEqual(metrics.semantic_matching_time, 0.0)
        self.assertEqual(metrics.embedding_generation_time, 0.0)
        self.assertEqual(metrics.faiss_search_time, 0.0)
        self.assertEqual(metrics.numeric_extraction_time, 0.0)
        self.assertEqual(metrics.association_time, 0.0)
        self.assertEqual(metrics.candidate_extraction_time, 0.0)

        self.assertEqual(metrics.total_segments, 0)
        self.assertEqual(metrics.total_terms_detected, 0)
        self.assertEqual(metrics.exact_matches, 0)
        self.assertEqual(metrics.semantic_matches, 0)
        self.assertEqual(metrics.numeric_associations, 0)

    def test_performance_metrics_with_values(self):
        """Test PerformanceMetrics with specific values."""
        metrics = PerformanceMetrics(
            total_processing_time=10.5,
            spacy_processing_time=2.0,
            exact_matching_time=1.5,
            semantic_matching_time=3.0,
            embedding_generation_time=1.0,
            faiss_search_time=0.5,
            numeric_extraction_time=0.8,
            association_time=0.7,
            candidate_extraction_time=1.0,
            total_segments=100,
            total_terms_detected=50,
            exact_matches=30,
            semantic_matches=15,
            numeric_associations=5
        )

        self.assertEqual(metrics.total_processing_time, 10.5)
        self.assertEqual(metrics.total_segments, 100)
        self.assertEqual(metrics.total_terms_detected, 50)

    def test_segments_per_second_calculation(self):
        """Test segments per second calculation."""
        # Normal case
        metrics = PerformanceMetrics(
            total_processing_time=5.0,
            total_segments=100
        )
        self.assertEqual(metrics.segments_per_second(), 20.0)

        # Zero time case (should return 0)
        metrics_zero = PerformanceMetrics(
            total_processing_time=0.0,
            total_segments=100
        )
        self.assertEqual(metrics_zero.segments_per_second(), 0)

        # Zero segments case
        metrics_no_segments = PerformanceMetrics(
            total_processing_time=5.0,
            total_segments=0
        )
        self.assertEqual(metrics_no_segments.segments_per_second(), 0.0)

    def test_terms_per_second_calculation(self):
        """Test terms per second calculation."""
        # Normal case
        metrics = PerformanceMetrics(
            total_processing_time=2.0,
            total_terms_detected=40
        )
        self.assertEqual(metrics.terms_per_second(), 20.0)

        # Zero time case (should return 0)
        metrics_zero = PerformanceMetrics(
            total_processing_time=0.0,
            total_terms_detected=40
        )
        self.assertEqual(metrics_zero.terms_per_second(), 0)

        # Zero terms case
        metrics_no_terms = PerformanceMetrics(
            total_processing_time=2.0,
            total_terms_detected=0
        )
        self.assertEqual(metrics_no_terms.terms_per_second(), 0.0)

    def test_performance_metrics_serialization(self):
        """Test PerformanceMetrics serialization to dict."""
        metrics = PerformanceMetrics(
            total_processing_time=5.5,
            spacy_processing_time=1.0,
            total_segments=50,
            exact_matches=25
        )

        metrics_dict = asdict(metrics)

        expected_keys = {
            'total_processing_time', 'spacy_processing_time', 'exact_matching_time',
            'semantic_matching_time', 'embedding_generation_time', 'faiss_search_time',
            'numeric_extraction_time', 'association_time', 'candidate_extraction_time',
            'total_segments', 'total_terms_detected', 'exact_matches',
            'semantic_matches', 'numeric_associations'
        }

        self.assertEqual(set(metrics_dict.keys()), expected_keys)
        self.assertEqual(metrics_dict['total_processing_time'], 5.5)
        self.assertEqual(metrics_dict['total_segments'], 50)

    def test_performance_metrics_field_types(self):
        """Test PerformanceMetrics field types."""
        metrics = PerformanceMetrics(
            total_processing_time=1.5,
            total_segments=10
        )

        # Test float fields
        self.assertIsInstance(metrics.total_processing_time, float)
        self.assertIsInstance(metrics.spacy_processing_time, float)
        self.assertIsInstance(metrics.exact_matching_time, float)

        # Test int fields
        self.assertIsInstance(metrics.total_segments, int)
        self.assertIsInstance(metrics.total_terms_detected, int)
        self.assertIsInstance(metrics.exact_matches, int)

    def test_performance_calculation_edge_cases(self):
        """Test edge cases in performance calculations."""
        # Very small processing time
        metrics = PerformanceMetrics(
            total_processing_time=0.001,
            total_segments=1,
            total_terms_detected=1
        )

        self.assertEqual(metrics.segments_per_second(), 1000.0)
        self.assertEqual(metrics.terms_per_second(), 1000.0)

        # Large numbers
        metrics_large = PerformanceMetrics(
            total_processing_time=1.0,
            total_segments=1000000,
            total_terms_detected=500000
        )

        self.assertEqual(metrics_large.segments_per_second(), 1000000.0)
        self.assertEqual(metrics_large.terms_per_second(), 500000.0)

    def test_performance_metrics_accumulation(self):
        """Test typical usage pattern of accumulating metrics."""
        metrics = PerformanceMetrics()

        # Simulate accumulating timing data
        metrics.spacy_processing_time += 1.0
        metrics.exact_matching_time += 0.5
        metrics.semantic_matching_time += 2.0
        metrics.total_processing_time = (
            metrics.spacy_processing_time +
            metrics.exact_matching_time +
            metrics.semantic_matching_time
        )

        # Simulate accumulating counts
        metrics.total_segments += 10
        metrics.exact_matches += 5
        metrics.semantic_matches += 3
        metrics.total_terms_detected = metrics.exact_matches + metrics.semantic_matches

        self.assertEqual(metrics.total_processing_time, 3.5)
        self.assertEqual(metrics.total_segments, 10)
        self.assertEqual(metrics.total_terms_detected, 8)
        self.assertAlmostEqual(metrics.segments_per_second(), 10/3.5, places=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)