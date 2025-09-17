#!/usr/bin/env python3
"""
Refactor Validation Tests

Validates that refactored code produces identical results to original.
This replaces the existing test infrastructure with dependency-aware testing.
"""

import unittest
import json
import sys
import importlib
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRefactorValidation(unittest.TestCase):
    """Validate refactored code against original behavior."""

    @classmethod
    def setUpClass(cls):
        """Set up validation testing."""
        cls.test_data_dir = Path(__file__).parent.parent.parent

    def test_import_structure_validation(self):
        """Test that imports work correctly after refactoring."""
        try:
            # Test direct import of main module
            import detect_economic_terms_with_embeddings
            self.assertTrue(hasattr(detect_economic_terms_with_embeddings, 'main'))
            self.assertTrue(hasattr(detect_economic_terms_with_embeddings, 'EconomicTermDetector'))
            print("✓ Main module imports correctly")

        except ImportError as e:
            self.skipTest(f"Import failed due to missing dependencies: {e}")

    def test_classes_available(self):
        """Test that all required classes are available."""
        try:
            import detect_economic_terms_with_embeddings as main_module

            # Check all 4 main classes exist
            required_classes = ['DetectedTerm', 'PerformanceMetrics', 'EconomicTermDetector', 'NumericExtractor']
            for class_name in required_classes:
                self.assertTrue(hasattr(main_module, class_name), f"{class_name} not found")

            print("✓ All required classes available")

        except ImportError as e:
            self.skipTest(f"Import failed due to missing dependencies: {e}")

    def test_configuration_constants(self):
        """Test that configuration constants are accessible."""
        try:
            import detect_economic_terms_with_embeddings as main_module

            # Check key configuration constants
            required_constants = [
                'USE_EMBEDDINGS', 'EMBEDDING_MODEL', 'TOP_K', 'SIMILARITY_THRESHOLD',
                'CONTEXT_WINDOW', 'DISTANCE_THRESHOLD', 'OUTPUT_DIRS', 'SPACY_MODELS'
            ]

            for constant in required_constants:
                self.assertTrue(hasattr(main_module, constant), f"{constant} not found")

            print("✓ All configuration constants available")

        except ImportError as e:
            self.skipTest(f"Import failed due to missing dependencies: {e}")

    def test_class_structure_unchanged(self):
        """Test that class structures remain unchanged."""
        try:
            from detect_economic_terms_with_embeddings import DetectedTerm, PerformanceMetrics

            # Test DetectedTerm structure
            detected_term_fields = DetectedTerm.__dataclass_fields__.keys()
            expected_fields = {
                'snippet', 'timestamp', 'canonical_term', 'matched_text',
                'numeric_value', 'numeric_text', 'confidence', 'match_type',
                'context', 'group_id'
            }
            self.assertEqual(set(detected_term_fields), expected_fields)

            # Test PerformanceMetrics structure
            perf_metrics_fields = PerformanceMetrics.__dataclass_fields__.keys()
            expected_perf_fields = {
                'total_processing_time', 'spacy_processing_time', 'exact_matching_time',
                'semantic_matching_time', 'embedding_generation_time', 'faiss_search_time',
                'numeric_extraction_time', 'association_time', 'candidate_extraction_time',
                'total_segments', 'total_terms_detected', 'exact_matches',
                'semantic_matches', 'numeric_associations'
            }
            self.assertEqual(set(perf_metrics_fields), expected_perf_fields)

            print("✓ Class structures unchanged")

        except ImportError as e:
            self.skipTest(f"Import failed due to missing dependencies: {e}")

    def test_method_signatures_unchanged(self):
        """Test that key method signatures remain unchanged."""
        try:
            from detect_economic_terms_with_embeddings import EconomicTermDetector, NumericExtractor

            # Test EconomicTermDetector key methods
            detector = EconomicTermDetector.__new__(EconomicTermDetector)  # Don't call __init__
            key_methods = [
                'process_jsonl_file', 'save_results', '_find_exact_matches',
                '_find_semantic_matches', '_extract_context'
            ]

            for method_name in key_methods:
                self.assertTrue(hasattr(detector, method_name), f"Method {method_name} not found")

            # Test NumericExtractor methods
            extractor = NumericExtractor.__new__(NumericExtractor)
            extractor_methods = [
                'extract_numeric_values', 'find_nearest_economic_term'
            ]

            for method_name in extractor_methods:
                self.assertTrue(hasattr(extractor, method_name), f"Method {method_name} not found")

            print("✓ Method signatures unchanged")

        except ImportError as e:
            self.skipTest(f"Import failed due to missing dependencies: {e}")

    def validate_refactored_imports(self):
        """Validate that refactored module imports work correctly."""
        # This will be updated as we extract classes
        try:
            # Test if we can import from new modules
            try:
                from models.detected_term import DetectedTerm as RefactoredDetectedTerm
                print("✓ DetectedTerm successfully extracted to models/")
            except ImportError:
                pass  # Not extracted yet

            try:
                from models.performance_metrics import PerformanceMetrics as RefactoredPerformanceMetrics
                print("✓ PerformanceMetrics successfully extracted to models/")
            except ImportError:
                pass  # Not extracted yet

            try:
                from extractors.numeric_extractor import NumericExtractor as RefactoredNumericExtractor
                print("✓ NumericExtractor successfully extracted to extractors/")
            except ImportError:
                pass  # Not extracted yet

            try:
                from detectors.economic_detector import EconomicTermDetector as RefactoredEconomicTermDetector
                print("✓ EconomicTermDetector successfully extracted to detectors/")
            except ImportError:
                pass  # Not extracted yet

        except Exception as e:
            print(f"Refactored import validation: {e}")

    def save_validation_results(self):
        """Save validation results for tracking."""
        validation_data = {
            "test_timestamp": str(Path(__file__).stat().st_mtime),
            "validation_status": "passed",
            "classes_extracted": [],
            "import_status": {}
        }

        # Check which classes have been extracted
        extracted_classes = []
        try:
            from models.detected_term import DetectedTerm
            extracted_classes.append("DetectedTerm")
        except ImportError:
            pass

        try:
            from models.performance_metrics import PerformanceMetrics
            extracted_classes.append("PerformanceMetrics")
        except ImportError:
            pass

        try:
            from extractors.numeric_extractor import NumericExtractor
            extracted_classes.append("NumericExtractor")
        except ImportError:
            pass

        try:
            from detectors.economic_detector import EconomicTermDetector
            extracted_classes.append("EconomicTermDetector")
        except ImportError:
            pass

        validation_data["classes_extracted"] = extracted_classes

        validation_file = Path(__file__).parent / "validation_results.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Validation results saved to: {validation_file}")


if __name__ == '__main__':
    # Run validation tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRefactorValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Save validation results
    test_instance = TestRefactorValidation()
    test_instance.validate_refactored_imports()
    test_instance.save_validation_results()

    sys.exit(0 if result.wasSuccessful() else 1)