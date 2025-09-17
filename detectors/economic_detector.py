#!/usr/bin/env python3
"""
EconomicTermDetector for advanced economic term detection

Advanced NLP pipeline for detecting economic indicators and technical terms in Spanish transcriptions.
Uses spaCy + PhraseMatcher for exact matches and SBERT + FAISS for semantic similarity.
"""

import json
import re
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import asdict
import uuid

import spacy
from spacy.matcher import PhraseMatcher

from models.detected_term import DetectedTerm
from models.performance_metrics import PerformanceMetrics
from extractors.numeric_extractor import NumericExtractor
from config.config_loader import ConfigLoader

# Setup logging
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers or faiss not available. Semantic matching disabled.")
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    faiss = None


class EconomicTermDetector:
    """Main class for economic term detection with embeddings."""

    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize the detector with models and configurations.

        Args:
            config: Configuration loader instance. If None, uses default config.
        """
        # Initialize configuration
        if config is None:
            from config.config_loader import get_config
            config = get_config()
        self.config = config

        # Get configuration values
        self.canonical_terms = config.get_canonical_terms()
        self.use_embeddings = config.use_embeddings
        self.embedding_model_name = config.embedding_model
        self.spacy_models = config.spacy_models
        self.similarity_threshold = config.similarity_threshold
        self.top_k = config.top_k
        self.context_window = config.context_window
        self.distance_threshold = config.distance_threshold
        self.output_dirs = config.get_output_dirs()

        # Initialize components
        self.nlp = self._load_spacy_model()
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.detected_terms = []
        self.canonical_embeddings = None
        self.faiss_index = None
        self.canonical_term_list = []

        # Initialize embeddings if available
        if self.use_embeddings and EMBEDDINGS_AVAILABLE:
            self.sentence_model = SentenceTransformer(self.embedding_model_name)
            self._prepare_embeddings()
        else:
            self.sentence_model = None
            logger.info("Semantic matching disabled - using exact matching only")

        self._setup_phrase_matcher()
        self.numeric_extractor = NumericExtractor(self.distance_threshold)
        self.metrics = PerformanceMetrics()

    def _load_spacy_model(self) -> spacy.Language:
        """Load Spanish spaCy model with robust fallback."""
        for model_name in self.spacy_models:
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

        for canonical_id, term_data in self.canonical_terms.items():
            # Create patterns for this canonical term
            patterns = [self.nlp(label.lower()) for label in term_data["labels"]]

            # Build direct lookup mapping for O(1) label validation
            for label in term_data["labels"]:
                self.label_to_canonical[label.lower()] = canonical_id

            # Register patterns under the canonical_id as the label
            self.phrase_matcher.add(canonical_id, patterns)
            total_patterns += len(patterns)

        logger.info(f"PhraseMatcher initialized with {total_patterns} patterns across {len(self.canonical_terms)} canonical terms")
        logger.info(f"Label lookup mapping created with {len(self.label_to_canonical)} entries")

    def _prepare_embeddings(self):
        """Prepare FAISS index with safe dtype handling."""
        if not self.sentence_model:
            return

        # Collect all canonical terms and their labels
        all_terms = []
        term_to_canonical = {}

        for canonical_id, term_data in self.canonical_terms.items():
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
                similarities, indices = self.faiss_index.search(embeddings, self.top_k)
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")
                return matches
            finally:
                self.metrics.faiss_search_time += time.time() - faiss_start

            for i, (candidate_text, start, end) in enumerate(filtered_candidates):
                best_similarity = similarities[i][0]
                best_idx = indices[i][0]

                if best_similarity >= self.similarity_threshold:
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
        context_start = max(0, start - self.context_window)
        context_end = min(len(text), end + self.context_window)

        context = text[context_start:context_end].strip()

        # Add ellipsis if context was truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."

        return context

    def _assign_group_ids(self):
        """Assign group IDs to detected terms for deduplication."""
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
                "embedding_model": self.embedding_model_name if self.use_embeddings else None,
                "embeddings_enabled": self.use_embeddings and EMBEDDINGS_AVAILABLE and self.sentence_model is not None,
                "similarity_threshold": self.similarity_threshold,
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
        json_path = Path(self.output_dirs['analysis']) / f"{base_filename}_detected_terms.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # Save Markdown
        md_path = Path(self.output_dirs['analysis']) / f"{base_filename}_detected_terms.md"
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
                term_info = self.canonical_terms.get(canonical_term, {})
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