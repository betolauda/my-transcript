#!/usr/bin/env python3
"""
Unit tests for DetectedTerm model

Tests the extracted DetectedTerm dataclass functionality.
"""

import unittest
import sys
import json
from pathlib import Path
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.detected_term import DetectedTerm


class TestDetectedTerm(unittest.TestCase):
    """Test DetectedTerm dataclass functionality."""

    def test_detected_term_creation(self):
        """Test basic DetectedTerm creation."""
        term = DetectedTerm(
            snippet="inflación",
            timestamp=1.0,
            canonical_term="inflacion",
            matched_text="inflación",
            numeric_value=25.5,
            numeric_text="25.5",
            confidence=1.0,
            match_type="exact",
            context="La inflación subió",
            group_id="test-123"
        )

        self.assertEqual(term.snippet, "inflación")
        self.assertEqual(term.timestamp, 1.0)
        self.assertEqual(term.canonical_term, "inflacion")
        self.assertEqual(term.matched_text, "inflación")
        self.assertEqual(term.numeric_value, 25.5)
        self.assertEqual(term.numeric_text, "25.5")
        self.assertEqual(term.confidence, 1.0)
        self.assertEqual(term.match_type, "exact")
        self.assertEqual(term.context, "La inflación subió")
        self.assertEqual(term.group_id, "test-123")

    def test_detected_term_defaults(self):
        """Test DetectedTerm with default values."""
        term = DetectedTerm(
            snippet="PIB",
            timestamp=2.0,
            canonical_term="pbi",
            matched_text="PIB",
            numeric_value=None,
            numeric_text=None,
            confidence=0.9,
            match_type="semantic",
            context="El PIB creció"
        )

        # group_id should default to empty string
        self.assertEqual(term.group_id, "")
        self.assertIsNone(term.numeric_value)
        self.assertIsNone(term.numeric_text)

    def test_detected_term_serialization(self):
        """Test DetectedTerm serialization to dict."""
        term = DetectedTerm(
            snippet="reservas",
            timestamp=3.0,
            canonical_term="reservas",
            matched_text="reservas internacionales",
            numeric_value=1000.0,
            numeric_text="1000 millones",
            confidence=0.8,
            match_type="numeric_association",
            context="Las reservas disminuyeron",
            group_id="grp-456"
        )

        term_dict = asdict(term)

        expected_keys = {
            'snippet', 'timestamp', 'canonical_term', 'matched_text',
            'numeric_value', 'numeric_text', 'confidence', 'match_type',
            'context', 'group_id'
        }

        self.assertEqual(set(term_dict.keys()), expected_keys)
        self.assertEqual(term_dict['snippet'], "reservas")
        self.assertEqual(term_dict['group_id'], "grp-456")

    def test_detected_term_json_serialization(self):
        """Test DetectedTerm JSON serialization."""
        term = DetectedTerm(
            snippet="dólar",
            timestamp=4.5,
            canonical_term="dolar",
            matched_text="dólar oficial",
            numeric_value=180.0,
            numeric_text="$180",
            confidence=1.0,
            match_type="exact",
            context="El dólar oficial cerró a $180",
            group_id=""
        )

        # Convert to dict and then to JSON
        term_dict = asdict(term)
        json_str = json.dumps(term_dict, ensure_ascii=False)

        # Parse back
        parsed = json.loads(json_str)

        self.assertEqual(parsed['snippet'], "dólar")
        self.assertEqual(parsed['timestamp'], 4.5)
        self.assertEqual(parsed['numeric_value'], 180.0)

    def test_detected_term_field_types(self):
        """Test DetectedTerm field type validation."""
        term = DetectedTerm(
            snippet="test",
            timestamp=1.0,
            canonical_term="test",
            matched_text="test",
            numeric_value=100.0,
            numeric_text="100",
            confidence=0.5,
            match_type="test",
            context="test context"
        )

        # Check types
        self.assertIsInstance(term.snippet, str)
        self.assertIsInstance(term.timestamp, float)
        self.assertIsInstance(term.canonical_term, str)
        self.assertIsInstance(term.matched_text, str)
        self.assertIsInstance(term.numeric_value, float)
        self.assertIsInstance(term.numeric_text, str)
        self.assertIsInstance(term.confidence, float)
        self.assertIsInstance(term.match_type, str)
        self.assertIsInstance(term.context, str)
        self.assertIsInstance(term.group_id, str)

    def test_detected_term_match_types(self):
        """Test different match types."""
        valid_match_types = ['exact', 'semantic', 'numeric_association']

        for match_type in valid_match_types:
            term = DetectedTerm(
                snippet="test",
                timestamp=1.0,
                canonical_term="test",
                matched_text="test",
                numeric_value=None,
                numeric_text=None,
                confidence=1.0,
                match_type=match_type,
                context="test"
            )
            self.assertEqual(term.match_type, match_type)

    def test_detected_term_equality(self):
        """Test DetectedTerm equality comparison."""
        term1 = DetectedTerm(
            snippet="inflación",
            timestamp=1.0,
            canonical_term="inflacion",
            matched_text="inflación",
            numeric_value=25.0,
            numeric_text="25%",
            confidence=1.0,
            match_type="exact",
            context="context",
            group_id="123"
        )

        term2 = DetectedTerm(
            snippet="inflación",
            timestamp=1.0,
            canonical_term="inflacion",
            matched_text="inflación",
            numeric_value=25.0,
            numeric_text="25%",
            confidence=1.0,
            match_type="exact",
            context="context",
            group_id="123"
        )

        term3 = DetectedTerm(
            snippet="PIB",
            timestamp=2.0,
            canonical_term="pbi",
            matched_text="PIB",
            numeric_value=None,
            numeric_text=None,
            confidence=0.9,
            match_type="semantic",
            context="different context",
            group_id="456"
        )

        self.assertEqual(term1, term2)
        self.assertNotEqual(term1, term3)


if __name__ == '__main__':
    unittest.main(verbosity=2)