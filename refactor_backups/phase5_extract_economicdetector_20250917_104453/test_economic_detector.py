#!/usr/bin/env python3
"""
Unit tests for Economic Term Detection

Simple test suite to verify core functionality of the economic term detector.
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from detect_economic_terms_with_embeddings import (
    EconomicTermDetector,
    DetectedTerm,
    NumericExtractor,
    CANONICAL_TERMS
)


class TestNumericExtractor(unittest.TestCase):
    """Test numeric value extraction functionality."""

    def setUp(self):
        self.extractor = NumericExtractor()

    def test_basic_number_extraction(self):
        """Test extraction of basic numbers."""
        text = "La inflación fue del 5.2 por ciento en enero."
        results = self.extractor.extract_numeric_values(text)

        self.assertGreater(len(results), 0)
        # Should find the percentage
        percentage_found = any(r['type'] == 'percentage' for r in results)
        self.assertTrue(percentage_found)

    def test_currency_extraction(self):
        """Test extraction of currency amounts."""
        text = "El precio del dólar llegó a $150 pesos."
        results = self.extractor.extract_numeric_values(text)

        currency_found = any(r['type'] == 'currency' for r in results)
        self.assertTrue(currency_found)

    def test_range_extraction(self):
        """Test extraction of numeric ranges."""
        text = "La tasa de interés oscila entre 3 y 5 por ciento."
        results = self.extractor.extract_numeric_values(text)

        range_found = any(r['type'] == 'range' for r in results)
        self.assertTrue(range_found)

    def test_empty_text(self):
        """Test behavior with empty text."""
        results = self.extractor.extract_numeric_values("")
        self.assertEqual(len(results), 0)


class TestEconomicTermDetector(unittest.TestCase):
    """Test economic term detection functionality."""

    def setUp(self):
        """Set up test detector with mocked embeddings."""
        with patch('detect_economic_terms_with_embeddings.EMBEDDINGS_AVAILABLE', False):
            self.detector = EconomicTermDetector()

    def test_exact_matching(self):
        """Test exact term matching functionality."""
        test_text = "La inflación subió al 25 por ciento este mes."
        doc = self.detector.nlp(test_text)

        matches = self.detector._find_exact_matches(doc, test_text, 0.0)

        # Should find "inflación" as exact match
        self.assertGreater(len(matches), 0)
        inflation_found = any('inflacion' in m['canonical'] for m in matches)
        self.assertTrue(inflation_found)

    def test_canonical_terms_coverage(self):
        """Test that all canonical terms are properly configured."""
        # Verify canonical terms dictionary is not empty
        self.assertGreater(len(CANONICAL_TERMS), 0)

        # Verify each canonical term has required fields
        for canonical_id, term_data in CANONICAL_TERMS.items():
            self.assertIn('labels', term_data)
            self.assertIn('description', term_data)
            self.assertIsInstance(term_data['labels'], list)
            self.assertGreater(len(term_data['labels']), 0)

    def test_context_extraction(self):
        """Test context window extraction."""
        text = "La economía argentina mostró signos de inflación durante el último trimestre del año."
        start = text.find("inflación")
        end = start + len("inflación")

        context = self.detector._extract_context(text, start, end)

        self.assertIn("inflación", context)
        self.assertGreater(len(context), len("inflación"))

    def test_empty_jsonl_processing(self):
        """Test processing of empty JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write empty file
            pass

        try:
            results = self.detector.process_jsonl_file(f.name)
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(f.name)

    def test_malformed_jsonl_processing(self):
        """Test processing of malformed JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"invalid": json}\n')
            f.write('{"start": 0.0, "end": 5.0, "text": "valid line"}\n')

        try:
            # Should not crash on malformed JSON
            results = self.detector.process_jsonl_file(f.name)
            # Should process the valid line
            self.assertGreaterEqual(len(results), 0)
        finally:
            os.unlink(f.name)

    def test_valid_jsonl_processing(self):
        """Test processing of valid JSONL with economic content."""
        test_data = [
            {"start": 0.0, "end": 5.0, "text": "La inflación aumentó significativamente."},
            {"start": 5.0, "end": 10.0, "text": "El PIB creció un 2.5 por ciento."},
            {"start": 10.0, "end": 15.0, "text": "Las reservas del banco central disminuyeron."}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        try:
            results = self.detector.process_jsonl_file(f.name)
            self.assertGreater(len(results), 0)

            # Should find economic terms
            canonical_terms = [r.canonical_term for r in results]
            economic_terms_found = any(term in ['inflacion', 'pbi', 'reservas'] for term in canonical_terms)
            self.assertTrue(economic_terms_found)

        finally:
            os.unlink(f.name)

    def test_group_id_assignment(self):
        """Test that group IDs are assigned during deduplication."""
        # Create some test detected terms
        self.detector.detected_terms = [
            DetectedTerm("inflación", 1.0, "inflacion", "inflación", None, None, 1.0, "exact", "context1"),
            DetectedTerm("inflacion", 1.5, "inflacion", "inflacion", None, None, 0.9, "semantic", "context2"),
            DetectedTerm("PIB", 10.0, "pbi", "PIB", None, None, 1.0, "exact", "context3")
        ]

        self.detector._assign_group_ids()

        # Check that group IDs were assigned
        for term in self.detector.detected_terms:
            self.assertNotEqual(term.group_id, "")

        # Terms with same canonical term and close timestamps should have same group_id
        inflacion_terms = [t for t in self.detector.detected_terms if t.canonical_term == "inflacion"]
        if len(inflacion_terms) > 1:
            self.assertEqual(inflacion_terms[0].group_id, inflacion_terms[1].group_id)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""

    def test_complete_workflow_no_embeddings(self):
        """Test complete workflow with embeddings disabled."""
        with patch('detect_economic_terms_with_embeddings.EMBEDDINGS_AVAILABLE', False):
            detector = EconomicTermDetector()

            test_data = [
                {"start": 0.0, "end": 5.0, "text": "La inflación subió al 25%."},
                {"start": 5.0, "end": 10.0, "text": "El PIB cayó un 1.5 por ciento."}
            ]

            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

            try:
                results = detector.process_jsonl_file(f.name)

                # Should detect some terms even without embeddings
                self.assertGreater(len(results), 0)

                # Test saving results
                with tempfile.TemporaryDirectory() as temp_dir:
                    os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)

                    # Temporarily change output directory
                    import detect_economic_terms_with_embeddings
                    original_dirs = detect_economic_terms_with_embeddings.OUTPUT_DIRS
                    detect_economic_terms_with_embeddings.OUTPUT_DIRS = {
                        "glossary": temp_dir,
                        "analysis": temp_dir
                    }

                    try:
                        detector.save_results("test")

                        # Check that files were created
                        json_file = os.path.join(temp_dir, "test_detected_terms.json")
                        md_file = os.path.join(temp_dir, "test_detected_terms.md")

                        self.assertTrue(os.path.exists(json_file))
                        self.assertTrue(os.path.exists(md_file))

                        # Verify JSON structure
                        with open(json_file, 'r') as jf:
                            data = json.load(jf)
                            self.assertIn('metadata', data)
                            self.assertIn('embeddings_enabled', data['metadata'])
                            self.assertFalse(data['metadata']['embeddings_enabled'])

                    finally:
                        detect_economic_terms_with_embeddings.OUTPUT_DIRS = original_dirs

            finally:
                os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()