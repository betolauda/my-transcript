#!/usr/bin/env python3
"""
DetectedTerm model for economic term detection

Data structure for detected economic terms with all metadata.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectedTerm:
    """Data structure for detected economic terms."""
    snippet: str
    timestamp: float
    canonical_term: str
    matched_text: str
    numeric_value: Optional[float]
    numeric_text: Optional[str]
    confidence: float
    match_type: str  # 'exact', 'semantic', 'numeric_association'
    context: str
    group_id: str = ""  # For deduplication grouping