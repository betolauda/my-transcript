#!/usr/bin/env python3
"""
NumericExtractor for economic term detection

Advanced numeric value extraction for Spanish text.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any

# Spanish number words for text extraction
NUM_WORDS = {
    "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "veinte": 20, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
    "cien": 100, "ciento": 100, "mil": 1000, "millón": 1000000, "millones": 1000000, "millon": 1000000
}

# Default distance threshold for term association
DEFAULT_DISTANCE_THRESHOLD = 10

logger = logging.getLogger(__name__)


class NumericExtractor:
    """Advanced numeric value extraction for Spanish text."""

    def __init__(self, distance_threshold: int = DEFAULT_DISTANCE_THRESHOLD):
        """Initialize regex patterns for various numeric formats."""
        self.distance_threshold = distance_threshold

        # Basic number patterns
        self.number_pattern = re.compile(r'\b\d+(?:[.,]\d+)?\b')

        # Percentage patterns
        self.percentage_patterns = [
            re.compile(r'\b(\d+(?:[.,]\d+)?)\s*(?:por\s*ciento|%)\b', re.IGNORECASE),
            re.compile(r'\b(\d+(?:[.,]\d+)?)\s*puntos?\s*(?:porcentuales?|básicos?)\b', re.IGNORECASE),
        ]

        # Currency patterns
        self.currency_patterns = [
            re.compile(r'\$\s*(\d+(?:[.,]\d+)?)\s*(?:millones?|millón|mm?)?\b', re.IGNORECASE),
            re.compile(r'u\$s?\s*(\d+(?:[.,]\d+)?)\s*(?:millones?|millón|mm?)?\b', re.IGNORECASE),
            re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:pesos?|dólares?|dolares?)\b', re.IGNORECASE),
        ]

        # Range patterns
        self.range_pattern = re.compile(
            r'\bentre\s+(\d+(?:[.,]\d+)?)\s*(?:%|por\s*ciento)?\s*y\s+(\d+(?:[.,]\d+)?)\s*(?:%|por\s*ciento)?\b',
            re.IGNORECASE
        )

        # Text number patterns
        self.text_number_pattern = re.compile(
            r'\b(?:' + '|'.join(NUM_WORDS.keys()) + r')\b',
            re.IGNORECASE
        )

    def extract_numeric_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract all numeric values from text with context."""
        results = []

        # Extract percentages
        for pattern in self.percentage_patterns:
            for match in pattern.finditer(text):
                value = self._normalize_number(match.group(1))
                if value is not None:
                    results.append({
                        'value': value,
                        'type': 'percentage',
                        'original_text': match.group(0),
                        'start': match.start(),
                        'end': match.end()
                    })

        # Extract currency amounts
        for pattern in self.currency_patterns:
            for match in pattern.finditer(text):
                value = self._normalize_number(match.group(1))
                if value is not None:
                    # Check for multipliers
                    full_match = match.group(0)
                    multiplier = 1
                    if re.search(r'millones?|millón|mm', full_match, re.IGNORECASE):
                        multiplier = 1000000

                    results.append({
                        'value': value * multiplier,
                        'type': 'currency',
                        'original_text': full_match,
                        'start': match.start(),
                        'end': match.end()
                    })

        # Extract ranges
        for match in self.range_pattern.finditer(text):
            start_val = self._normalize_number(match.group(1))
            end_val = self._normalize_number(match.group(2))
            if start_val is not None and end_val is not None:
                results.append({
                    'value': (start_val, end_val),
                    'type': 'range',
                    'original_text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

        # Extract basic numbers
        for match in self.number_pattern.finditer(text):
            # Skip if already captured by other patterns
            if not any(r['start'] <= match.start() < r['end'] for r in results):
                value = self._normalize_number(match.group(0))
                if value is not None:
                    results.append({
                        'value': value,
                        'type': 'number',
                        'original_text': match.group(0),
                        'start': match.start(),
                        'end': match.end()
                    })

        # Extract text numbers
        text_numbers = self._extract_text_numbers(text)
        results.extend(text_numbers)

        return sorted(results, key=lambda x: x['start'])

    def _normalize_number(self, text: str) -> Optional[float]:
        """Convert string number to float with context-aware separator detection."""
        try:
            # Clean input
            text = text.strip()
            if not text:
                return None

            # Handle simple cases first
            if ',' not in text and '.' not in text:
                return float(text)

            # Context-aware separator detection
            if ',' in text and '.' in text:
                # Both separators present - determine which is decimal
                comma_pos = text.rfind(',')
                dot_pos = text.rfind('.')

                if dot_pos > comma_pos:
                    # Format: 1.234,56 or 1,234.56 - dot comes after comma
                    if dot_pos - comma_pos <= 3:
                        # Likely European format: 1.234,56 (dot=thousands, comma=decimal)
                        normalized = text.replace('.', '').replace(',', '.')
                    else:
                        # Likely US format with error or unusual spacing
                        normalized = text.replace(',', '')
                else:
                    # Format: 1,234.56 (comma=thousands, dot=decimal)
                    normalized = text.replace(',', '')
            elif ',' in text:
                # Only comma present
                comma_count = text.count(',')
                comma_pos = text.rfind(',')

                if comma_count == 1:
                    # Check if it's decimal separator (Spanish style) or thousands
                    digits_after_comma = len(text) - comma_pos - 1
                    if digits_after_comma <= 3 and comma_pos > 0:
                        # Likely decimal separator: 12,34
                        normalized = text.replace(',', '.')
                    else:
                        # Likely thousands separator: 1,234
                        normalized = text.replace(',', '')
                else:
                    # Multiple commas - thousands separators: 1,234,567
                    normalized = text.replace(',', '')
            else:
                # Only dot present - assume it's decimal separator
                normalized = text

            return float(normalized)

        except (ValueError, AttributeError):
            logger.debug(f"Failed to normalize number: {text}")
            return None

    def _extract_text_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers written as text with precise character position recovery."""
        results = []
        words = re.findall(r'\b\w+\b', text.lower())

        i = 0
        while i < len(words):
            # Look for number patterns like "tres punto dos"
            if words[i] in NUM_WORDS:
                start_word = words[i]
                current_value = NUM_WORDS[start_word]
                original_start = i
                i += 1

                # Handle decimal points
                if i < len(words) - 2 and words[i] == 'punto':
                    decimal_part = 0
                    decimal_pos = 0.1
                    i += 1

                    while i < len(words) and words[i] in NUM_WORDS:
                        decimal_part += NUM_WORDS[words[i]] * decimal_pos
                        decimal_pos /= 10
                        i += 1

                    current_value += decimal_part

                # Check for percentage indicator
                value_type = 'number'
                if i < len(words) and words[i] in ['por', 'porciento']:
                    if i + 1 < len(words) and words[i + 1] == 'ciento':
                        value_type = 'percentage'
                        i += 2
                    elif words[i] == 'porciento':
                        value_type = 'percentage'
                        i += 1

                # Reconstruct original text phrase
                phrase = ' '.join(words[original_start:i])

                # Find character positions in original text
                start_pos, end_pos = self._find_phrase_positions(text, phrase)

                results.append({
                    'value': current_value,
                    'type': value_type,
                    'original_text': phrase,
                    'start': start_pos,
                    'end': end_pos
                })
            else:
                i += 1

        return results

    def _find_phrase_positions(self, text: str, phrase: str) -> Tuple[int, int]:
        """Find character positions of phrase in original text with fuzzy matching."""
        # Try exact match first (case-insensitive)
        text_lower = text.lower()
        phrase_lower = phrase.lower()

        pos = text_lower.find(phrase_lower)
        if pos != -1:
            return pos, pos + len(phrase)

        # If exact match fails, try word-boundary matching
        # Split phrase into words and find approximate positions
        phrase_words = phrase_lower.split()
        if not phrase_words:
            return -1, -1

        # Find first word position
        first_word = phrase_words[0]
        pattern = r'\b' + re.escape(first_word) + r'\b'
        match = re.search(pattern, text_lower)

        if match:
            start_pos = match.start()

            # Try to find end position by looking for last word
            if len(phrase_words) > 1:
                last_word = phrase_words[-1]
                # Search for last word after first word position
                last_pattern = r'\b' + re.escape(last_word) + r'\b'
                last_match = re.search(last_pattern, text_lower[start_pos:])

                if last_match:
                    end_pos = start_pos + last_match.end()
                    return start_pos, end_pos

            # Fallback: estimate end position
            estimated_length = len(phrase) + 10  # Add buffer for spacing differences
            end_pos = min(start_pos + estimated_length, len(text))
            return start_pos, end_pos

        # Ultimate fallback - return -1 for impossible cases
        logger.debug(f"Could not find position for text number phrase: '{phrase}'")
        return -1, -1

    def find_nearest_economic_term(self, numeric_position: int, economic_terms: List[Tuple[int, int, str]],
                                   max_distance: Optional[int] = None) -> Optional[str]:
        """Find the nearest economic term to a numeric value with improved distance calculation."""
        if max_distance is None:
            max_distance = self.distance_threshold

        # Skip if position is invalid
        if numeric_position < 0:
            logger.debug(f"Skipping numeric association: invalid position {numeric_position}")
            return None

        closest_term = None
        min_distance = float('inf')

        for start, end, term in economic_terms:
            # Skip terms with invalid positions
            if start < 0 or end < 0:
                continue

            # Calculate distance considering overlap and proximity
            if numeric_position >= start and numeric_position <= end:
                # Numeric value is inside the term span - very close
                distance = 0
            elif numeric_position < start:
                # Numeric value is before the term
                distance = start - numeric_position
            else:
                # Numeric value is after the term
                distance = numeric_position - end

            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_term = term

        if closest_term:
            logger.debug(f"Associated numeric at position {numeric_position} with term '{closest_term}' (distance: {min_distance})")

        return closest_term