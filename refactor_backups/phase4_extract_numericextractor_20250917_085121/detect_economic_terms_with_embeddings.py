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
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict
from collections import defaultdict, Counter

import spacy
from spacy.matcher import PhraseMatcher

# Configuration constants
USE_EMBEDDINGS = True
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.75
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

# Import models from models package
from models.detected_term import DetectedTerm
from models.performance_metrics import PerformanceMetrics

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
        self.metrics = PerformanceMetrics()

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
        """Process JSONL file and detect economic terms with performance tracking."""
        self.detected_terms = []
        self.metrics = PerformanceMetrics()  # Reset metrics for new file

        processing_start_time = time.time()

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

        # Finalize metrics
        self.metrics.total_processing_time = time.time() - processing_start_time
        self.metrics.total_terms_detected = len(self.detected_terms)

        logger.info(f"Processed file: {len(self.detected_terms)} terms detected")
        self._log_performance_summary()
        return self.detected_terms

    def _process_text_segment(self, text: str, timestamp: float):
        """Process a single text segment for economic terms with performance tracking."""
        segment_start_time = time.time()

        # Process with spaCy (timed)
        spacy_start = time.time()
        doc = self.nlp(text)
        self.metrics.spacy_processing_time += time.time() - spacy_start

        # Extract numeric values first (timed)
        numeric_start = time.time()
        numeric_values = self.numeric_extractor.extract_numeric_values(text)
        self.metrics.numeric_extraction_time += time.time() - numeric_start

        # Phase 1: Exact matching with PhraseMatcher (timed)
        exact_start = time.time()
        exact_matches = self._find_exact_matches(doc, text, timestamp)
        self.metrics.exact_matching_time += time.time() - exact_start

        # Phase 2: Semantic matching (if enabled) (timed)
        semantic_matches = []
        if self.sentence_model and self.faiss_index:
            semantic_start = time.time()
            semantic_matches = self._find_semantic_matches(doc, text, timestamp, exact_matches)
            self.metrics.semantic_matching_time += time.time() - semantic_start

        # Combine all matches
        all_matches = exact_matches + semantic_matches

        # Phase 3: Associate numeric values with nearby terms (timed)
        association_start = time.time()
        self._associate_numeric_values(numeric_values, all_matches, text, timestamp)
        self.metrics.association_time += time.time() - association_start

        # Update segment metrics
        self.metrics.total_segments += 1
        segment_time = time.time() - segment_start_time

        # Count matches for this segment
        segment_exact = len(exact_matches)
        segment_semantic = len(semantic_matches)

        # Count numeric values that have nearby economic terms
        term_positions = [(m['start'], m['end'], m['canonical']) for m in all_matches]
        segment_numeric = len([nv for nv in numeric_values
                             if self.numeric_extractor.find_nearest_economic_term(nv['start'], term_positions)])

        self.metrics.exact_matches += segment_exact
        self.metrics.semantic_matches += segment_semantic
        self.metrics.numeric_associations += segment_numeric

        logger.debug(f"Segment processed in {segment_time:.3f}s: {segment_exact} exact, {segment_semantic} semantic, {segment_numeric} numeric")

    def _log_performance_summary(self):
        """Log comprehensive performance summary and diagnostics."""
        m = self.metrics

        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Total processing time: {m.total_processing_time:.3f}s")
        logger.info(f"Segments processed: {m.total_segments}")
        logger.info(f"Terms detected: {m.total_terms_detected}")
        logger.info(f"Throughput: {m.segments_per_second():.2f} segments/sec, {m.terms_per_second():.2f} terms/sec")

        logger.info("=== TIMING BREAKDOWN ===")
        if m.total_processing_time > 0:
            logger.info(f"spaCy processing: {m.spacy_processing_time:.3f}s ({100*m.spacy_processing_time/m.total_processing_time:.1f}%)")
            logger.info(f"Exact matching: {m.exact_matching_time:.3f}s ({100*m.exact_matching_time/m.total_processing_time:.1f}%)")
            logger.info(f"Semantic matching: {m.semantic_matching_time:.3f}s ({100*m.semantic_matching_time/m.total_processing_time:.1f}%)")
            logger.info(f"  - Candidate extraction: {m.candidate_extraction_time:.3f}s ({100*m.candidate_extraction_time/m.total_processing_time:.1f}%)")
            logger.info(f"  - Embedding generation: {m.embedding_generation_time:.3f}s ({100*m.embedding_generation_time/m.total_processing_time:.1f}%)")
            logger.info(f"  - FAISS search: {m.faiss_search_time:.3f}s ({100*m.faiss_search_time/m.total_processing_time:.1f}%)")
            logger.info(f"Numeric extraction: {m.numeric_extraction_time:.3f}s ({100*m.numeric_extraction_time/m.total_processing_time:.1f}%)")
            logger.info(f"Association: {m.association_time:.3f}s ({100*m.association_time/m.total_processing_time:.1f}%)")

        logger.info("=== DETECTION BREAKDOWN ===")
        logger.info(f"Exact matches: {m.exact_matches}")
        logger.info(f"Semantic matches: {m.semantic_matches}")
        logger.info(f"Numeric associations: {m.numeric_associations}")

        # Performance warnings
        if m.segments_per_second() < 1.0:
            logger.warning("Low throughput detected (<1 segment/sec). Consider optimization.")
        if m.semantic_matching_time > 0.5 * m.total_processing_time:
            logger.warning("Semantic matching is >50% of processing time. Consider reducing candidates.")

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
                context=context,
                group_id=""  # Will be assigned during deduplication
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

        # Time candidate extraction
        candidate_start = time.time()

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

        # If no candidates found (e.g., blank model), extract precise sliding window n-grams
        if not candidates:
            logger.debug("No noun chunks or entities found, using precise sliding window extraction")
            candidates.extend(self._extract_sliding_window_ngrams(doc, text))

        # Always add frequent phrases for better coverage
        frequent_phrases = self._extract_frequent_phrases(doc, text)
        if frequent_phrases:
            logger.debug(f"Adding {len(frequent_phrases)} frequent phrase candidates")
            candidates.extend(frequent_phrases)

        # Add pattern-based economic phrases
        pattern_phrases = self._extract_economic_patterns(text)
        if pattern_phrases:
            logger.debug(f"Adding {len(pattern_phrases)} pattern-based economic phrases")
            candidates.extend(pattern_phrases)

        # Remove candidates that overlap with exact matches
        filtered_candidates = []
        for candidate_text, start, end in candidates:
            overlap = any(
                not (end <= match['start'] or start >= match['end'])
                for match in exact_matches
            )
            if not overlap:
                filtered_candidates.append((candidate_text, start, end))

        self.metrics.candidate_extraction_time += time.time() - candidate_start

        # Process candidates with semantic similarity
        if filtered_candidates:
            candidate_texts = [cand[0] for cand in filtered_candidates]

            # Time embedding generation
            embedding_start = time.time()
            embeddings = self.sentence_model.encode(candidate_texts)
            self.metrics.embedding_generation_time += time.time() - embedding_start

            # Ensure float32 dtype BEFORE normalization
            embeddings = embeddings.astype('float32')

            # Safe normalization on known dtype
            try:
                faiss.normalize_L2(embeddings)
            except Exception as e:
                logger.warning(f"FAISS normalization failed: {e}")
                return matches

            # Time FAISS search
            faiss_start = time.time()
            try:
                similarities, indices = self.faiss_index.search(embeddings, TOP_K)
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")
                return matches
            finally:
                self.metrics.faiss_search_time += time.time() - faiss_start

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
                        context=context,
                        group_id=""  # Will be assigned during deduplication
                    )

                    self.detected_terms.append(detected_term)
                    matches.append({
                        'start': start,
                        'end': end,
                        'text': candidate_text,
                        'canonical': canonical_term
                    })

        return matches

    def _extract_sliding_window_ngrams(self, doc, text: str, max_candidates: int = 200) -> List[Tuple[str, int, int]]:
        """Extract overlapping n-grams with precise character positions using sliding window."""
        candidates = []

        # Filter meaningful tokens (alphabetic, not too short, not stopwords)
        meaningful_tokens = []
        for token in doc:
            if (token.is_alpha and
                len(token.text) > 2 and
                not token.is_stop and
                not token.is_punct):
                meaningful_tokens.append(token)

            # Early termination for very large documents
            if len(meaningful_tokens) > 500:
                logger.debug("Limiting meaningful tokens to 500 for performance")
                break

        # Extract overlapping n-grams (1-4 tokens) with limits
        for window_size in range(1, 5):  # 1-gram to 4-gram
            window_candidates = 0
            for i in range(len(meaningful_tokens) - window_size + 1):
                # Early termination to prevent excessive candidates
                if len(candidates) >= max_candidates:
                    logger.debug(f"Reached max candidates ({max_candidates}) for sliding window extraction")
                    break

                # Limit candidates per window size to maintain variety
                if window_candidates >= max_candidates // 4:
                    break

                # Get token span
                start_token = meaningful_tokens[i]
                end_token = meaningful_tokens[i + window_size - 1]

                # Calculate precise character positions
                start_char = start_token.idx
                end_char = end_token.idx + len(end_token.text)

                # Extract phrase text from original text using positions
                phrase_text = text[start_char:end_char].lower().strip()

                # Validate phrase quality
                if (len(phrase_text) > 2 and
                    len(phrase_text.split()) <= 4 and
                    not phrase_text.isdigit()):
                    candidates.append((phrase_text, start_char, end_char))
                    window_candidates += 1

            # Early exit if we have enough candidates
            if len(candidates) >= max_candidates:
                break

        # Remove duplicates while preserving order and positions
        seen = set()
        unique_candidates = []
        for phrase, start, end in candidates:
            key = (phrase, start, end)
            if key not in seen:
                seen.add(key)
                unique_candidates.append((phrase, start, end))

        logger.debug(f"Extracted {len(unique_candidates)} sliding window n-gram candidates")
        return unique_candidates

    def _extract_frequent_phrases(self, doc, text: str, min_frequency: int = 2) -> List[Tuple[str, int, int]]:
        """Extract frequent multi-word phrases that might indicate economic concepts."""

        phrase_counts = Counter()
        phrase_positions = {}  # Track first occurrence positions

        # Extract meaningful token sequences
        meaningful_tokens = [token for token in doc if
                           token.is_alpha and len(token.text) > 2 and not token.is_stop]

        # Count bi-grams and tri-grams
        for window_size in [2, 3]:
            for i in range(len(meaningful_tokens) - window_size + 1):
                tokens_in_window = meaningful_tokens[i:i + window_size]
                phrase = " ".join(token.text.lower() for token in tokens_in_window)

                # Skip if contains numbers or is too short
                if any(char.isdigit() for char in phrase) or len(phrase) < 5:
                    continue

                phrase_counts[phrase] += 1

                # Store position of first occurrence
                if phrase not in phrase_positions:
                    start_char = tokens_in_window[0].idx
                    end_char = tokens_in_window[-1].idx + len(tokens_in_window[-1].text)
                    phrase_positions[phrase] = (start_char, end_char)

        # Filter frequent phrases
        frequent_candidates = []
        for phrase, count in phrase_counts.items():
            if count >= min_frequency:
                start_char, end_char = phrase_positions[phrase]
                frequent_candidates.append((phrase, start_char, end_char))

        # Sort by frequency (most frequent first)
        frequent_candidates.sort(key=lambda x: phrase_counts[x[0]], reverse=True)

        # Limit to top 20 to avoid noise
        return frequent_candidates[:20]

    def _extract_economic_patterns(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract economic phrases using pattern matching for Spanish economic terminology."""
        patterns = [
            # Rate/percentage patterns
            r'\btasa\s+de\s+\w+(?:\s+\w+)*\b',
            r'\bíndice\s+de\s+\w+(?:\s+\w+)*\b',
            r'\btipo\s+de\s+cambio\b',
            r'\btasa\s+de\s+interés\b',
            r'\btasa\s+de\s+inflación\b',

            # Economic indicators
            r'\bproducto\s+bruto\s+interno\b',
            r'\bproducto\s+interno\s+bruto\b',
            r'\bbalanza\s+comercial\b',
            r'\bbalanza\s+de\s+pagos\b',
            r'\bdeuda\s+externa\b',
            r'\bdeuda\s+pública\b',

            # Central bank and monetary policy
            r'\bbanco\s+central\b',
            r'\bpolítica\s+monetaria\b',
            r'\bpolítica\s+fiscal\b',
            r'\breservas\s+internacionales\b',
            r'\bbase\s+monetaria\b',

            # Market terminology
            r'\bmercado\s+de\s+capitales\b',
            r'\bmercado\s+cambiario\b',
            r'\bmercado\s+financiero\b',
            r'\bbolsa\s+de\s+valores\b',
            r'\briesgo\s+país\b',

            # Crisis and economic conditions
            r'\bcrisis\s+económica\b',
            r'\bcrisis\s+financiera\b',
            r'\brecesión\s+económica\b',
            r'\bcrecimiento\s+económico\b',

            # Specific economic measures
            r'\bcepo\s+cambiario\b',
            r'\bbrecha\s+cambiaria\b',
            r'\bdólar\s+blue\b',
            r'\bdólar\s+oficial\b',
            r'\bfondo\s+monetario\s+internacional\b',
        ]

        economic_candidates = []
        text_lower = text.lower()

        for pattern in patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                phrase = match.group(0).strip()
                start_pos = match.start()
                end_pos = match.end()

                # Validate phrase quality
                if (len(phrase) > 3 and
                    len(phrase.split()) >= 2 and  # At least two words
                    not phrase.isdigit()):
                    economic_candidates.append((phrase, start_pos, end_pos))

        # Remove duplicates
        seen = set()
        unique_patterns = []
        for phrase, start, end in economic_candidates:
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                unique_patterns.append((phrase, start, end))

        return unique_patterns

    def _associate_numeric_values(self, numeric_values: List[Dict], term_matches: List[Dict],
                                  text: str, timestamp: float):
        """Associate numeric values with nearby economic terms."""
        # Create list of term positions for distance calculation
        term_positions = [(match['start'], match['end'], match['canonical']) for match in term_matches]

        for numeric in numeric_values:
            # Find nearest economic term (now handles position validation internally)
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
                    context=context,
                    group_id=""  # Will be assigned during deduplication
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

    def _assign_group_ids(self):
        """Assign group IDs to detected terms for deduplication."""
        import uuid

        # Sort terms by timestamp for grouping
        sorted_terms = sorted(self.detected_terms, key=lambda x: x.timestamp)

        groups = []
        for term in sorted_terms:
            # Find existing group within proximity threshold
            assigned_group = None
            for group in groups:
                # Check if term is within character proximity of any term in the group
                for group_term in group:
                    if (term.canonical_term == group_term.canonical_term and
                        abs(term.timestamp - group_term.timestamp) <= 5.0):  # 5 second window
                        assigned_group = group
                        break
                if assigned_group:
                    break

            if assigned_group:
                assigned_group.append(term)
                term.group_id = assigned_group[0].group_id
            else:
                # Create new group
                group_id = str(uuid.uuid4())[:8]  # Short UUID
                term.group_id = group_id
                groups.append([term])

    def save_results(self, base_filename: str):
        """Save detected terms in JSON and Markdown formats."""
        # Assign group IDs for deduplication
        self._assign_group_ids()

        # Prepare data for JSON export
        results_data = {
            "metadata": {
                "total_terms": len(self.detected_terms),
                "embedding_model": EMBEDDING_MODEL if USE_EMBEDDINGS else None,
                "embeddings_enabled": USE_EMBEDDINGS and EMBEDDINGS_AVAILABLE and self.sentence_model is not None,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "exact_matches": len([t for t in self.detected_terms if t.match_type == 'exact']),
                "semantic_matches": len([t for t in self.detected_terms if t.match_type == 'semantic']),
                "numeric_associations": len([t for t in self.detected_terms if t.match_type == 'numeric_association'])
            },
            "performance_metrics": {
                "total_processing_time": self.metrics.total_processing_time,
                "total_segments": self.metrics.total_segments,
                "segments_per_second": self.metrics.segments_per_second(),
                "terms_per_second": self.metrics.terms_per_second(),
                "timing_breakdown": {
                    "spacy_processing_time": self.metrics.spacy_processing_time,
                    "exact_matching_time": self.metrics.exact_matching_time,
                    "semantic_matching_time": self.metrics.semantic_matching_time,
                    "candidate_extraction_time": self.metrics.candidate_extraction_time,
                    "embedding_generation_time": self.metrics.embedding_generation_time,
                    "faiss_search_time": self.metrics.faiss_search_time,
                    "numeric_extraction_time": self.metrics.numeric_extraction_time,
                    "association_time": self.metrics.association_time
                }
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
                                   max_distance: int = DISTANCE_THRESHOLD) -> Optional[str]:
        """Find the nearest economic term to a numeric value with improved distance calculation."""
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