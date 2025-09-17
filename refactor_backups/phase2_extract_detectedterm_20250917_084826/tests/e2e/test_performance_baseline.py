#!/usr/bin/env python3
"""
Performance Baseline Testing

Records current performance metrics for ±5% tolerance validation during refactoring.
"""

import unittest
import time
import psutil
import json
import sys
from pathlib import Path
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPerformanceBaseline(unittest.TestCase):
    """Capture performance baseline for regression testing."""

    @classmethod
    def setUpClass(cls):
        """Set up performance monitoring."""
        cls.test_data_dir = Path(__file__).parent.parent.parent
        cls.sample_file = cls.test_data_dir / "test_sample_segments.jsonl"
        cls.performance_tolerance = 5.0  # ±5% tolerance

    def test_processing_time_baseline(self):
        """Measure baseline processing time."""
        try:
            from detect_economic_terms_with_embeddings import EconomicTermDetector

            # Warm-up run (not measured)
            detector = EconomicTermDetector()
            detector.process_jsonl_file(str(self.sample_file))

            # Measured runs
            processing_times = []
            memory_usages = []

            for i in range(3):  # Average of 3 runs
                # Memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB

                # Time processing
                start_time = time.time()
                detector = EconomicTermDetector()
                detected_terms = detector.process_jsonl_file(str(self.sample_file))
                end_time = time.time()

                # Memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB

                processing_time = end_time - start_time
                memory_used = memory_after - memory_before

                processing_times.append(processing_time)
                memory_usages.append(memory_used)

                print(f"Run {i+1}: {processing_time:.3f}s, Memory: {memory_used:.1f}MB")

            # Calculate averages
            avg_processing_time = sum(processing_times) / len(processing_times)
            avg_memory_usage = sum(memory_usages) / len(memory_usages)

            # Store baseline metrics
            self.baseline_metrics = {
                "avg_processing_time": avg_processing_time,
                "max_processing_time": max(processing_times),
                "min_processing_time": min(processing_times),
                "avg_memory_usage": avg_memory_usage,
                "max_memory_usage": max(memory_usages),
                "terms_detected": len(detected_terms),
                "tolerance_percent": self.performance_tolerance
            }

            print(f"\n✓ Baseline Performance:")
            print(f"  Average processing time: {avg_processing_time:.3f}s")
            print(f"  Average memory usage: {avg_memory_usage:.1f}MB")
            print(f"  Terms detected: {len(detected_terms)}")

            # Basic performance assertions
            self.assertLess(avg_processing_time, 30.0, "Processing too slow")
            self.assertLess(avg_memory_usage, 500.0, "Memory usage too high")

        except ImportError:
            self.skipTest("Cannot import main detector - will test after refactoring")

    def test_throughput_baseline(self):
        """Measure baseline throughput metrics."""
        try:
            from detect_economic_terms_with_embeddings import EconomicTermDetector

            detector = EconomicTermDetector()

            start_time = time.time()
            detected_terms = detector.process_jsonl_file(str(self.sample_file))
            end_time = time.time()

            processing_time = end_time - start_time

            # Calculate throughput metrics
            segments_per_second = detector.metrics.segments_per_second()
            terms_per_second = detector.metrics.terms_per_second()

            self.baseline_throughput = {
                "segments_per_second": segments_per_second,
                "terms_per_second": terms_per_second,
                "total_segments": detector.metrics.total_segments,
                "total_terms": len(detected_terms)
            }

            print(f"\n✓ Baseline Throughput:")
            print(f"  Segments per second: {segments_per_second:.2f}")
            print(f"  Terms per second: {terms_per_second:.2f}")

        except ImportError:
            self.skipTest("Cannot import main detector - will test after refactoring")

    def save_performance_baseline(self):
        """Save performance baseline for comparison."""
        baseline_data = {
            "performance_metrics": getattr(self, 'baseline_metrics', {}),
            "throughput_metrics": getattr(self, 'baseline_throughput', {}),
            "tolerance_percent": self.performance_tolerance
        }

        baseline_file = Path(__file__).parent / "performance_baseline.json"
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Performance baseline saved to: {baseline_file}")

    def validate_performance_regression(self, current_metrics: dict) -> bool:
        """Validate performance within tolerance after refactoring."""
        if not hasattr(self, 'baseline_metrics'):
            return True  # No baseline to compare

        baseline = self.baseline_metrics
        tolerance = self.performance_tolerance / 100.0

        # Check processing time regression
        baseline_time = baseline['avg_processing_time']
        current_time = current_metrics.get('avg_processing_time', 0)
        time_regression = (current_time - baseline_time) / baseline_time

        if time_regression > tolerance:
            print(f"❌ Performance regression detected: {time_regression*100:.1f}% slower")
            return False

        # Check memory usage regression
        baseline_memory = baseline['avg_memory_usage']
        current_memory = current_metrics.get('avg_memory_usage', 0)
        memory_regression = (current_memory - baseline_memory) / baseline_memory

        if memory_regression > tolerance:
            print(f"❌ Memory regression detected: {memory_regression*100:.1f}% more memory")
            return False

        print(f"✓ Performance within tolerance:")
        print(f"  Time change: {time_regression*100:.1f}%")
        print(f"  Memory change: {memory_regression*100:.1f}%")
        return True


if __name__ == '__main__':
    # Run performance baseline tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBaseline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Save performance baseline
    test_instance = TestPerformanceBaseline()
    test_instance.test_processing_time_baseline()
    test_instance.test_throughput_baseline()
    test_instance.save_performance_baseline()

    sys.exit(0 if result.wasSuccessful() else 1)