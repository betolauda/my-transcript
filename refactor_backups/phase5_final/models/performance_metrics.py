#!/usr/bin/env python3
"""
PerformanceMetrics model for economic term detection

Performance and timing metrics for the detection process.
"""

from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance and timing metrics for the detection process."""
    total_processing_time: float = 0.0
    spacy_processing_time: float = 0.0
    exact_matching_time: float = 0.0
    semantic_matching_time: float = 0.0
    embedding_generation_time: float = 0.0
    faiss_search_time: float = 0.0
    numeric_extraction_time: float = 0.0
    association_time: float = 0.0
    candidate_extraction_time: float = 0.0

    total_segments: int = 0
    total_terms_detected: int = 0
    exact_matches: int = 0
    semantic_matches: int = 0
    numeric_associations: int = 0

    def segments_per_second(self) -> float:
        """Calculate processing throughput in segments per second."""
        return self.total_segments / self.total_processing_time if self.total_processing_time > 0 else 0

    def terms_per_second(self) -> float:
        """Calculate detection throughput in terms per second."""
        return self.total_terms_detected / self.total_processing_time if self.total_processing_time > 0 else 0