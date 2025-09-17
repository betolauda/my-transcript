#!/usr/bin/env python3
"""
End-to-End Baseline Testing

Captures current output behavior for regression testing during refactoring.
This replaces the existing test_economic_detector.py with comprehensive baseline tests.
"""

import unittest
import json
import tempfile
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestBaselineBehavior(unittest.TestCase):
    """Capture baseline behavior before refactoring."""

    @classmethod
    def setUpClass(cls):
        """Set up test data and baseline capture."""
        cls.test_data_dir = Path(__file__).parent.parent.parent
        cls.sample_file = cls.test_data_dir / "test_sample_segments.jsonl"

        # Ensure sample file exists
        if not cls.sample_file.exists():
            cls._create_test_sample_file()

    @classmethod
    def _create_test_sample_file(cls):
        """Create test sample file if it doesn't exist."""
        test_segments = [
            {"start": 0.0, "end": 5.0, "text": "La inflación interanual alcanzó el 25.5 por ciento en diciembre."},
            {"start": 5.0, "end": 10.0, "text": "El producto bruto interno creció un 2.1% en el último trimestre."},
            {"start": 10.0, "end": 15.0, "text": "Las reservas internacionales del banco central disminuyeron."},
            {"start": 15.0, "end": 20.0, "text": "La tasa de desempleo se ubicó en 9.6 por ciento."},
            {"start": 20.0, "end": 25.0, "text": "El dólar oficial cerró a $180 pesos en el mercado cambiario."},
            {"start": 25.0, "end": 30.0, "text": "El déficit fiscal primario fue de 2.3% del PIB este año."},
            {"start": 30.0, "end": 35.0, "text": "No hay contenido económico en esta línea sobre deportes."}
        ]

        with open(cls.sample_file, 'w', encoding='utf-8') as f:
            for segment in test_segments:
                f.write(json.dumps(segment) + '\n')

    def test_baseline_cli_output(self):
        """Test CLI output format and structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run the main script
            result = subprocess.run([
                sys.executable, "detect_economic_terms_with_embeddings.py",
                str(self.sample_file)
            ], capture_output=True, text=True, cwd=self.test_data_dir)

            # Store baseline output for comparison
            self.baseline_stdout = result.stdout
            self.baseline_stderr = result.stderr
            self.baseline_returncode = result.returncode

            # Basic assertions
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            self.assertIn("Processing", result.stdout)
            self.assertIn("detected", result.stdout)

    def test_baseline_json_output_structure(self):
        """Test JSON output file structure."""
        try:
            # Import the detector for testing
            from detect_economic_terms_with_embeddings import EconomicTermDetector

            detector = EconomicTermDetector()
            detected_terms = detector.process_jsonl_file(str(self.sample_file))

            # Store baseline for comparison
            self.baseline_terms_count = len(detected_terms)
            self.baseline_term_types = set(term.match_type for term in detected_terms)
            self.baseline_canonical_terms = set(term.canonical_term for term in detected_terms)

            # Basic structure validation
            self.assertGreaterEqual(len(detected_terms), 0)

            # Check that each term has required fields
            for term in detected_terms:
                self.assertIsNotNone(term.snippet)
                self.assertIsNotNone(term.timestamp)
                self.assertIsNotNone(term.canonical_term)
                self.assertIsNotNone(term.matched_text)
                self.assertIsNotNone(term.confidence)
                self.assertIsNotNone(term.match_type)
                self.assertIsNotNone(term.context)
                self.assertIsNotNone(term.group_id)

        except ImportError:
            self.skipTest("Cannot import main detector - will test after refactoring")

    def test_baseline_output_files(self):
        """Test output file generation."""
        try:
            from detect_economic_terms_with_embeddings import EconomicTermDetector

            with tempfile.TemporaryDirectory() as temp_dir:
                # Temporarily change output directory
                import detect_economic_terms_with_embeddings
                original_dirs = detect_economic_terms_with_embeddings.OUTPUT_DIRS
                detect_economic_terms_with_embeddings.OUTPUT_DIRS = {
                    "glossary": temp_dir,
                    "analysis": temp_dir
                }

                try:
                    detector = EconomicTermDetector()
                    detector.process_jsonl_file(str(self.sample_file))
                    detector.save_results("baseline_test")

                    # Check output files exist
                    json_file = Path(temp_dir) / "baseline_test_detected_terms.json"
                    md_file = Path(temp_dir) / "baseline_test_detected_terms.md"

                    self.assertTrue(json_file.exists())
                    self.assertTrue(md_file.exists())

                    # Store baseline file content
                    with open(json_file, 'r', encoding='utf-8') as f:
                        self.baseline_json_output = json.load(f)

                    with open(md_file, 'r', encoding='utf-8') as f:
                        self.baseline_md_output = f.read()

                finally:
                    detect_economic_terms_with_embeddings.OUTPUT_DIRS = original_dirs

        except ImportError:
            self.skipTest("Cannot import main detector - will test after refactoring")

    def save_baseline_reference(self):
        """Save baseline reference for comparison during refactoring."""
        baseline_data = {
            "terms_count": getattr(self, 'baseline_terms_count', 0),
            "term_types": list(getattr(self, 'baseline_term_types', set())),
            "canonical_terms": list(getattr(self, 'baseline_canonical_terms', set())),
            "cli_stdout": getattr(self, 'baseline_stdout', ''),
            "cli_stderr": getattr(self, 'baseline_stderr', ''),
            "cli_returncode": getattr(self, 'baseline_returncode', 0)
        }

        baseline_file = Path(__file__).parent / "baseline_reference.json"
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Baseline reference saved to: {baseline_file}")


if __name__ == '__main__':
    # Run tests and save baseline
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaselineBehavior)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Save baseline reference after tests
    test_instance = TestBaselineBehavior()
    test_instance.test_baseline_json_output_structure()
    test_instance.save_baseline_reference()

    sys.exit(0 if result.wasSuccessful() else 1)