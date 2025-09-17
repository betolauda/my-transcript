#!/usr/bin/env python3
"""
Economic Term Detection with Embeddings

Advanced NLP pipeline for detecting economic indicators and technical terms in Spanish transcriptions.
Uses spaCy + PhraseMatcher for exact matches and SBERT + FAISS for semantic similarity.

Author: AI Engineer Senior
Date: 2025-01-17
"""

import json
import sys
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import spacy
from spacy.matcher import PhraseMatcher

# Configuration constants
USE_EMBEDDINGS = True
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.60
CONTEXT_WINDOW = 20
DISTANCE_THRESHOLD = 10
OUTPUT_DIRS = {"glossary": "glossary", "analysis": "outputs"}
SPACY_MODELS = ["es_core_news_trf", "es_core_news_md", "es_core_news_sm"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers or faiss not available. Semantic matching disabled.")
    EMBEDDINGS_AVAILABLE = False

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
    match_type: str  # 'exact', 'semantic'
    context: str

# ---------- Canonical seed dictionary ----------
CANONICAL_TERMS = {
    "inflacion": {
        "labels": ["inflación", "inflacion", "IPC", "índice de precios", "suba de precios",
                  "suba de los precios", "inflation", "price index", "consumer price index"],
        "description": "Inflation – general rise of prices, often measured by IPC."
    },
    "pbi": {
        "labels": ["PIB", "pbi", "producto bruto interno", "producto bruto",
                  "producto interno bruto", "GDP", "gross domestic product"],
        "description": "Gross Domestic Product."
    },
    "reservas": {
        "labels": ["reservas", "reservas internacionales", "reservas del banco central",
                  "reserves", "international reserves", "central bank reserves"],
        "description": "International reserves held by central bank."
    },
    "desempleo": {
        "labels": ["desempleo", "tasa de desempleo", "paro", "unemployment", "unemployment rate"],
        "description": "Unemployment rate."
    },
    "dolar": {
        "labels": ["dólar", "dolar", "u$s", "usd", "dolares", "dólares", "tipo de cambio",
                  "dollar", "exchange rate", "USD"],
        "description": "US Dollar / exchange rate references."
    },
    "peso": {
        "labels": ["peso", "pesos", "$", "ARS", "peso argentino"],
        "description": "Argentine peso references."
    },
    "deficit": {
        "labels": ["déficit", "deficit", "déficit fiscal", "deficit fiscal", "fiscal deficit"],
        "description": "Fiscal deficit."
    },
    "superavit": {
        "labels": ["superávit", "superavit", "superávit fiscal", "fiscal surplus"],
        "description": "Fiscal surplus."
    },
    "tasa_interes": {
        "labels": ["tasa de interés", "tasa de interes", "tasa", "tasa activa", "tasa pasiva",
                  "tasa de política monetaria", "interest rate", "policy rate", "benchmark rate"],
        "description": "Interest rate."
    },
    "m2": {
        "labels": ["M2", "m2", "m1", "M1", "oferta monetaria", "monetary supply", "money supply"],
        "description": "Monetary aggregates."
    },
    "importaciones": {
        "labels": ["importaciones", "volumen de importaciones", "importe de importaciones",
                  "imports", "import volume"],
        "description": "Imports volume."
    },
    "subsidios": {
        "labels": ["subsidios", "subsidio", "subsidies", "subsidy"],
        "description": "Subsidies."
    },
}

# ---------- Numeric extraction helpers ----------
NUM_WORDS = {
    "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "veinte": 20, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
    "cien": 100, "ciento": 100, "mil": 1000, "millón": 1000000, "millones": 1000000, "millon": 1000000
}

class EconomicTermDetector:
    """Main class for economic term detection with embeddings."""

    def __init__(self):
        """Initialize the detector with models and configurations."""
        self.nlp = self._load_spacy_model()
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.detected_terms = []
        self.canonical_embeddings = None
        self.faiss_index = None
        self.canonical_term_list = []

        # Initialize embeddings if available
        if USE_EMBEDDINGS and EMBEDDINGS_AVAILABLE:
            self.sentence_model = SentenceTransformer(EMBEDDING_MODEL)
            self._prepare_embeddings()
        else:
            self.sentence_model = None
            logger.info("Semantic matching disabled - using exact matching only")

        self._setup_phrase_matcher()
        self.numeric_extractor = NumericExtractor()

    def _load_spacy_model(self) -> spacy.Language:
        """Load Spanish spaCy model with robust fallback."""
        for model_name in SPACY_MODELS:
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                continue

        # Fallback to blank model with warning
        logger.warning("No trained spaCy models found. Falling back to blank Spanish model.")
        logger.warning("For better accuracy, install: python -m spacy download es_core_news_sm")

        nlp = spacy.blank("es")

        # Add minimal pipeline components for basic functionality
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        logger.info("Initialized blank Spanish model with sentencizer")
        return nlp

    def _setup_phrase_matcher(self):
        """Setup PhraseMatcher with per-canonical-term patterns and lookup mapping."""
        total_patterns = 0
        self.label_to_canonical = {}  # Direct lookup mapping for exact matches

        for canonical_id, term_data in CANONICAL_TERMS.items():
            # Create patterns for this canonical term
            patterns = [self.nlp(label.lower()) for label in term_data["labels"]]

            # Build direct lookup mapping for O(1) label validation
            for label in term_data["labels"]:
                self.label_to_canonical[label.lower()] = canonical_id

            # Register patterns under the canonical_id as the label
            self.phrase_matcher.add(canonical_id, patterns)
            total_patterns += len(patterns)

        logger.info(f"PhraseMatcher initialized with {total_patterns} patterns across {len(CANONICAL_TERMS)} canonical terms")
        logger.info(f"Label lookup mapping created with {len(self.label_to_canonical)} entries")

    def _prepare_embeddings(self):
        """Prepare FAISS index with safe dtype handling."""
        if not self.sentence_model:
            return

        # Collect all canonical terms and their labels
        all_terms = []
        term_to_canonical = {}

        for canonical_id, term_data in CANONICAL_TERMS.items():
            for label in term_data["labels"]:
                all_terms.append(label.lower())
                term_to_canonical[label.lower()] = canonical_id

        # Generate embeddings
        embeddings = self.sentence_model.encode(all_terms)

        # Ensure float32 dtype BEFORE normalization
        embeddings = embeddings.astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Safe normalization on known dtype
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)  # Already float32

        self.canonical_term_list = all_terms
        self.term_to_canonical = term_to_canonical

        logger.info(f"FAISS index initialized with {len(all_terms)} canonical terms")

    def process_jsonl_file(self, file_path: str) -> List[DetectedTerm]:
        """Process JSONL file and detect economic terms."""
        self.detected_terms = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                        text = record.get('text', '')
                        timestamp = record.get('start', 0.0)

                        if text:
                            self._process_text_segment(text, timestamp)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

        logger.info(f"Processed file: {len(self.detected_terms)} terms detected")
        return self.detected_terms

    def _process_text_segment(self, text: str, timestamp: float):
        """Process a single text segment for economic terms."""
        # Process with spaCy
        doc = self.nlp(text)

        # Extract numeric values first
        numeric_values = self.numeric_extractor.extract_numeric_values(text)

        # Phase 1: Exact matching with PhraseMatcher
        exact_matches = self._find_exact_matches(doc, text, timestamp)

        # Phase 2: Semantic matching (if enabled)
        semantic_matches = []
        if self.sentence_model and self.faiss_index:
            semantic_matches = self._find_semantic_matches(doc, text, timestamp, exact_matches)

        # Combine all matches
        all_matches = exact_matches + semantic_matches

        # Phase 3: Associate numeric values with nearby terms
        self._associate_numeric_values(numeric_values, all_matches, text, timestamp)

    def _find_exact_matches(self, doc, text: str, timestamp: float) -> List[Dict]:
        """Find exact matches using PhraseMatcher with direct canonical lookup."""
        matches = []
        phrase_matches = self.phrase_matcher(doc)

        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            matched_text = span.text.lower()

            # Direct O(1) lookup: match_id → canonical_term
            canonical_term = self.nlp.vocab.strings[match_id]

            context = self._extract_context(text, span.start_char, span.end_char)

            detected_term = DetectedTerm(
                snippet=matched_text,
                timestamp=timestamp,
                canonical_term=canonical_term,
                matched_text=matched_text,
                numeric_value=None,
                numeric_text=None,
                confidence=1.0,
                match_type='exact',
                context=context
            )

            self.detected_terms.append(detected_term)
            matches.append({
                'start': span.start_char,
                'end': span.end_char,
                'text': matched_text,
                'canonical': canonical_term
            })

        return matches

    def _find_semantic_matches(self, doc, text: str, timestamp: float, exact_matches: List[Dict]) -> List[Dict]:
        """Find semantic matches using SBERT + FAISS."""
        matches = []

        # Extract candidate phrases (noun chunks, entities)
        candidates = []

        # Add noun chunks (requires dependency parsing)
        try:
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Limit phrase length
                    candidates.append((chunk.text.lower(), chunk.start_char, chunk.end_char))
        except Exception as e:
            # noun_chunks requires dependency parsing - skip for blank models
            if "noun_chunks requires the dependency parse" in str(e) or "[E029]" in str(e):
                logger.debug("Skipping noun_chunks extraction: dependency parsing not available")
            else:
                raise

        # Add named entities (blank models may have limited NER)
        try:
            for ent in doc.ents:
                if len(ent.text.split()) <= 4:
                    candidates.append((ent.text.lower(), ent.start_char, ent.end_char))
        except Exception as e:
            logger.debug(f"Named entity extraction failed: {e}")

        # If no candidates found (e.g., blank model), extract simple token n-grams as fallback
        if not candidates:
            logger.debug("No noun chunks or entities found, using token-based fallback")
            tokens = [token.text.lower() for token in doc if token.is_alpha and len(token.text) > 2]
            for i in range(len(tokens) - 1):
                for n in range(1, min(4, len(tokens) - i + 1)):  # 1-3 gram
                    if i + n <= len(tokens):
                        phrase = " ".join(tokens[i:i+n])
                        # Approximate character positions (best effort)
                        start_char = i * 6  # rough estimate
                        end_char = start_char + len(phrase)
                        candidates.append((phrase, start_char, end_char))

        # Remove candidates that overlap with exact matches
        filtered_candidates = []
        for candidate_text, start, end in candidates:
            overlap = any(
                not (end <= match['start'] or start >= match['end'])
                for match in exact_matches
            )
            if not overlap:
                filtered_candidates.append((candidate_text, start, end))

        # Process candidates with semantic similarity
        if filtered_candidates:
            candidate_texts = [cand[0] for cand in filtered_candidates]
            embeddings = self.sentence_model.encode(candidate_texts)

            # Ensure float32 dtype BEFORE normalization
            embeddings = embeddings.astype('float32')

            # Safe normalization on known dtype
            import faiss
            faiss.normalize_L2(embeddings)

            # Search for similar terms
            similarities, indices = self.faiss_index.search(embeddings, TOP_K)

            for i, (candidate_text, start, end) in enumerate(filtered_candidates):
                best_similarity = similarities[i][0]
                best_idx = indices[i][0]

                if best_similarity >= SIMILARITY_THRESHOLD:
                    matched_canonical_label = self.canonical_term_list[best_idx]
                    canonical_term = self.term_to_canonical[matched_canonical_label]

                    context = self._extract_context(text, start, end)

                    detected_term = DetectedTerm(
                        snippet=candidate_text,
                        timestamp=timestamp,
                        canonical_term=canonical_term,
                        matched_text=candidate_text,
                        numeric_value=None,
                        numeric_text=None,
                        confidence=float(best_similarity),
                        match_type='semantic',
                        context=context
                    )

                    self.detected_terms.append(detected_term)
                    matches.append({
                        'start': start,
                        'end': end,
                        'text': candidate_text,
                        'canonical': canonical_term
                    })

        return matches

    def _associate_numeric_values(self, numeric_values: List[Dict], term_matches: List[Dict],
                                  text: str, timestamp: float):
        """Associate numeric values with nearby economic terms."""
        # Create list of term positions for distance calculation
        term_positions = [(match['start'], match['end'], match['canonical']) for match in term_matches]

        for numeric in numeric_values:
            if numeric['start'] < 0:  # Skip text numbers without positions
                continue

            # Find nearest economic term
            nearest_term = self.numeric_extractor.find_nearest_economic_term(
                numeric['start'], term_positions
            )

            if nearest_term:
                context = self._extract_context(text, numeric['start'], numeric['end'])

                detected_term = DetectedTerm(
                    snippet=numeric['original_text'],
                    timestamp=timestamp,
                    canonical_term=nearest_term,
                    matched_text=f"[{numeric['type']}] {numeric['original_text']}",
                    numeric_value=numeric['value'],
                    numeric_text=numeric['original_text'],
                    confidence=0.8,  # Lower confidence for associated values
                    match_type='numeric_association',
                    context=context
                )

                self.detected_terms.append(detected_term)

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context window around detected term."""
        context_start = max(0, start - CONTEXT_WINDOW)
        context_end = min(len(text), end + CONTEXT_WINDOW)

        context = text[context_start:context_end].strip()

        # Add ellipsis if context was truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."

        return context

    def save_results(self, base_filename: str):
        """Save detected terms in JSON and Markdown formats."""
        # Prepare data for JSON export
        results_data = {
            "metadata": {
                "total_terms": len(self.detected_terms),
                "embedding_model": EMBEDDING_MODEL if USE_EMBEDDINGS else None,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "exact_matches": len([t for t in self.detected_terms if t.match_type == 'exact']),
                "semantic_matches": len([t for t in self.detected_terms if t.match_type == 'semantic']),
                "numeric_associations": len([t for t in self.detected_terms if t.match_type == 'numeric_association'])
            },
            "detected_terms": [asdict(term) for term in self.detected_terms]
        }

        # Save JSON
        json_path = Path(OUTPUT_DIRS['analysis']) / f"{base_filename}_detected_terms.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # Save Markdown
        md_path = Path(OUTPUT_DIRS['analysis']) / f"{base_filename}_detected_terms.md"
        self._save_markdown_report(md_path, results_data)

        logger.info(f"Results saved to {json_path} and {md_path}")

    def _save_markdown_report(self, file_path: Path, data: Dict):
        """Generate structured Markdown report."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# Economic Terms Detection Report\n\n")

            # Metadata section
            metadata = data['metadata']
            f.write("## Summary\n\n")
            f.write(f"- **Total terms detected**: {metadata['total_terms']}\n")
            f.write(f"- **Exact matches**: {metadata['exact_matches']}\n")
            f.write(f"- **Semantic matches**: {metadata['semantic_matches']}\n")
            f.write(f"- **Numeric associations**: {metadata['numeric_associations']}\n")
            if metadata['embedding_model']:
                f.write(f"- **Embedding model**: {metadata['embedding_model']}\n")
                f.write(f"- **Similarity threshold**: {metadata['similarity_threshold']}\n")
            f.write("\n")

            # Group by canonical term
            terms_by_canonical = defaultdict(list)
            for term in data['detected_terms']:
                terms_by_canonical[term['canonical_term']].append(term)

            f.write("## Detected Terms by Category\n\n")
            for canonical_term, terms in sorted(terms_by_canonical.items()):
                term_info = CANONICAL_TERMS.get(canonical_term, {})
                description = term_info.get('description', 'No description available')

                f.write(f"### {canonical_term.replace('_', ' ').title()}\n")
                f.write(f"*{description}*\n\n")

                f.write("| Timestamp | Text | Type | Confidence | Numeric Value | Context |\n")
                f.write("|-----------|------|------|------------|---------------|----------|\n")

                for term in sorted(terms, key=lambda x: x['timestamp']):
                    numeric_display = str(term['numeric_value']) if term['numeric_value'] is not None else "-"
                    context_short = term['context'][:50] + "..." if len(term['context']) > 50 else term['context']

                    f.write(f"| {term['timestamp']:.1f}s | {term['matched_text']} | {term['match_type']} | "
                           f"{term['confidence']:.2f} | {numeric_display} | {context_short} |\n")

                f.write(f"\n**Total occurrences**: {len(terms)}\n\n")

            # Timeline section
            f.write("## Timeline\n\n")
            f.write("| Timestamp | Term | Type | Text | Numeric Value |\n")
            f.write("|-----------|------|------|------|---------------|\n")

            for term in sorted(data['detected_terms'], key=lambda x: x['timestamp']):
                numeric_display = str(term['numeric_value']) if term['numeric_value'] is not None else "-"
                f.write(f"| {term['timestamp']:.1f}s | {term['canonical_term']} | {term['match_type']} | "
                       f"{term['matched_text']} | {numeric_display} |\n")


class NumericExtractor:
    """Advanced numeric value extraction for Spanish text."""

    def __init__(self):
        """Initialize regex patterns for various numeric formats."""
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
        """Convert string number to float, handling Spanish formats."""
        try:
            # Replace comma decimal separator with dot
            normalized = text.replace(',', '.')
            return float(normalized)
        except ValueError:
            return None

    def _extract_text_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers written as text (e.g., 'tres punto dos por ciento')."""
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

                # Reconstruct original text
                original_text = ' '.join(words[original_start:i])

                results.append({
                    'value': current_value,
                    'type': value_type,
                    'original_text': original_text,
                    'start': -1,  # Text position would need more complex tracking
                    'end': -1
                })
            else:
                i += 1

        return results

    def find_nearest_economic_term(self, numeric_position: int, economic_terms: List[Tuple[int, int, str]],
                                   max_distance: int = DISTANCE_THRESHOLD) -> Optional[str]:
        """Find the nearest economic term to a numeric value."""
        if numeric_position < 0:  # Skip text numbers without positions
            return None

        closest_term = None
        min_distance = float('inf')

        for start, end, term in economic_terms:
            # Calculate distance (consider both directions)
            distance = min(
                abs(numeric_position - end),    # Distance from end of term
                abs(start - numeric_position)   # Distance from start of term
            )

            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_term = term

        return closest_term


def main():
    """Main function to process JSONL file and detect economic terms."""
    if len(sys.argv) != 2:
        print("Usage: python detect_economic_terms_with_embeddings.py <input.jsonl>")
        print("\nExample:")
        print("  python detect_economic_terms_with_embeddings.py outputs/S08E05.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    # Create output directories
    for directory in OUTPUT_DIRS.values():
        os.makedirs(directory, exist_ok=True)

    # Extract base filename for output files
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    print(f"Processing {input_file}...")
    print(f"Configuration:")
    print(f"  - Embeddings enabled: {USE_EMBEDDINGS and EMBEDDINGS_AVAILABLE}")
    print(f"  - Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"  - Context window: {CONTEXT_WINDOW} characters")
    print()

    # Initialize detector
    try:
        detector = EconomicTermDetector()
        logger.info("Economic term detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        print(f"Error: Failed to initialize detector: {e}")
        sys.exit(1)

    # Process the file
    try:
        detected_terms = detector.process_jsonl_file(input_file)

        if not detected_terms:
            print("Warning: No economic terms detected in the input file")
            print("This could indicate:")
            print("  - The audio doesn't contain economic content")
            print("  - The similarity threshold is too high")
            print("  - The canonical terms dictionary needs expansion")
        else:
            print(f"✓ Successfully detected {len(detected_terms)} economic terms")

            # Print summary by type
            exact_count = len([t for t in detected_terms if t.match_type == 'exact'])
            semantic_count = len([t for t in detected_terms if t.match_type == 'semantic'])
            numeric_count = len([t for t in detected_terms if t.match_type == 'numeric_association'])

            print(f"  - Exact matches: {exact_count}")
            print(f"  - Semantic matches: {semantic_count}")
            print(f"  - Numeric associations: {numeric_count}")

            # Show top canonical terms
            from collections import Counter
            canonical_counts = Counter(term.canonical_term for term in detected_terms)
            print(f"\nTop detected categories:")
            for canonical, count in canonical_counts.most_common(5):
                print(f"  - {canonical.replace('_', ' ').title()}: {count} occurrences")

        # Save results
        detector.save_results(base_filename)

        print(f"\n✓ Results saved to:")
        print(f"  - JSON: {OUTPUT_DIRS['analysis']}/{base_filename}_detected_terms.json")
        print(f"  - Markdown: {OUTPUT_DIRS['analysis']}/{base_filename}_detected_terms.md")

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        print(f"Error: Failed to process file: {e}")
        sys.exit(1)

    print("\nEconomic term detection completed successfully!")


if __name__ == "__main__":
    main()