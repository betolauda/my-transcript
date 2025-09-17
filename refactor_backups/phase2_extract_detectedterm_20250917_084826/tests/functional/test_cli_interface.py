#!/usr/bin/env python3
"""
CLI Interface Testing

Validates that command-line interface remains unchanged during refactoring.
Replaces existing test files with comprehensive CLI validation.
"""

import unittest
import subprocess
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCLIInterface(unittest.TestCase):
    """Test CLI interface remains unchanged during refactoring."""

    @classmethod
    def setUpClass(cls):
        """Set up CLI testing."""
        cls.test_data_dir = Path(__file__).parent.parent.parent
        cls.main_script = cls.test_data_dir / "detect_economic_terms_with_embeddings.py"
        cls.sample_file = cls.test_data_dir / "test_sample_segments.jsonl"

    def test_cli_help_unchanged(self):
        """Test that help text remains unchanged."""
        result = subprocess.run([
            sys.executable, str(self.main_script)
        ], capture_output=True, text=True, cwd=self.test_data_dir)

        # Should show usage information
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stdout)
        self.assertIn("detect_economic_terms_with_embeddings.py", result.stdout)

    def test_cli_file_processing(self):
        """Test CLI file processing functionality."""
        if not self.sample_file.exists():
            self.skipTest("Sample file not available")

        result = subprocess.run([
            sys.executable, str(self.main_script), str(self.sample_file)
        ], capture_output=True, text=True, cwd=self.test_data_dir)

        # Should process successfully
        self.assertEqual(result.returncode, 0, f"CLI processing failed: {result.stderr}")
        self.assertIn("Processing", result.stdout)
        self.assertIn("detected", result.stdout)

    def test_cli_nonexistent_file(self):
        """Test CLI error handling for nonexistent files."""
        result = subprocess.run([
            sys.executable, str(self.main_script), "nonexistent.jsonl"
        ], capture_output=True, text=True, cwd=self.test_data_dir)

        # Should fail gracefully
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stdout)

    def test_cli_output_format(self):
        """Test CLI output format consistency."""
        if not self.sample_file.exists():
            self.skipTest("Sample file not available")

        result = subprocess.run([
            sys.executable, str(self.main_script), str(self.sample_file)
        ], capture_output=True, text=True, cwd=self.test_data_dir)

        # Check output format elements
        self.assertIn("Configuration:", result.stdout)
        self.assertIn("Embeddings enabled:", result.stdout)
        self.assertIn("Similarity threshold:", result.stdout)
        self.assertIn("Results saved to:", result.stdout)

    def test_cli_exit_codes(self):
        """Test CLI exit codes are consistent."""
        # Success case
        if self.sample_file.exists():
            result = subprocess.run([
                sys.executable, str(self.main_script), str(self.sample_file)
            ], capture_output=True, text=True, cwd=self.test_data_dir)
            self.assertEqual(result.returncode, 0)

        # Error case - no arguments
        result = subprocess.run([
            sys.executable, str(self.main_script)
        ], capture_output=True, text=True, cwd=self.test_data_dir)
        self.assertEqual(result.returncode, 1)

        # Error case - nonexistent file
        result = subprocess.run([
            sys.executable, str(self.main_script), "nonexistent.jsonl"
        ], capture_output=True, text=True, cwd=self.test_data_dir)
        self.assertEqual(result.returncode, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)