#!/usr/bin/env python3
"""
Similarity Threshold Tuning Script

Diagnostic tool to analyze semantic similarity scores and tune the similarity threshold
for corpus-specific performance. Helps identify optimal threshold values by showing
nearest canonical terms for sample economic segments.

Usage:
    python tune_similarity_threshold.py [--threshold 0.75] [--samples sample_segments.json]
"""

import argparse
import json
import sys
import csv
from typing import List, Dict, Tuple
import logging

# Import the detector components
from detect_economic_terms_with_embeddings import (
    EconomicTermDetector,
    CANONICAL_TERMS,
    SIMILARITY_THRESHOLD,
    EMBEDDINGS_AVAILABLE,
    USE_EMBEDDINGS
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThresholdTuner:
    """Utility for analyzing and tuning similarity thresholds."""

    def __init__(self):
        """Initialize the tuner with embedding model."""
        if not (USE_EMBEDDINGS and EMBEDDINGS_AVAILABLE):
            logger.error("Embeddings are not available. Cannot tune threshold without embeddings.")
            sys.exit(1)

        self.detector = EconomicTermDetector()
        if not self.detector.sentence_model:
            logger.error("Failed to initialize sentence model. Cannot proceed.")
            sys.exit(1)

        logger.info(f"Initialized tuner with embedding model: {self.detector.sentence_model}")

    def analyze_sample_segments(self, sample_segments: List[Dict]) -> List[Dict]:
        """Analyze similarity scores for sample segments."""
        results = []

        for i, segment in enumerate(sample_segments):
            text = segment.get('text', '')
            expected_terms = segment.get('expected_terms', [])

            logger.info(f"Analyzing segment {i+1}/{len(sample_segments)}: '{text[:50]}...'")

            # Process text to get candidates
            doc = self.detector.nlp(text)
            similarities = self._get_similarity_analysis(text, doc)

            result = {
                'segment_id': i + 1,
                'text': text,
                'expected_terms': expected_terms,
                'similarity_analysis': similarities
            }
            results.append(result)

        return results

    def _get_similarity_analysis(self, text: str, doc) -> List[Dict]:
        """Get detailed similarity analysis for text candidates."""
        # Extract candidates using the same logic as the detector
        candidates = []

        # Add noun chunks if available
        try:
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:
                    candidates.append(chunk.text.lower())
        except Exception:
            pass

        # Add named entities if available
        try:
            for ent in doc.ents:
                if len(ent.text.split()) <= 4:
                    candidates.append(ent.text.lower())
        except Exception:
            pass

        # If no candidates, use sliding window
        if not candidates:
            # Simple word extraction for tuning
            words = text.lower().split()
            for i in range(len(words)):
                for j in range(i+1, min(i+4, len(words)+1)):
                    phrase = ' '.join(words[i:j])
                    if len(phrase) > 3:
                        candidates.append(phrase)

        # Remove duplicates
        candidates = list(set(candidates))

        # Get embeddings and similarities
        if not candidates:
            return []

        try:
            candidate_embeddings = self.detector.sentence_model.encode(candidates)
            candidate_embeddings = candidate_embeddings.astype('float32')

            # Normalize
            import faiss
            faiss.normalize_L2(candidate_embeddings)

            # Search against canonical terms
            similarities, indices = self.detector.faiss_index.search(candidate_embeddings, 5)  # Top 5

            analysis_results = []
            for i, candidate in enumerate(candidates):
                top_matches = []
                for j in range(min(5, len(similarities[i]))):
                    if similarities[i][j] > 0:  # Only positive similarities
                        canonical_label = self.detector.canonical_term_list[indices[i][j]]
                        canonical_term = self.detector.term_to_canonical[canonical_label]
                        top_matches.append({
                            'canonical_term': canonical_term,
                            'canonical_label': canonical_label,
                            'similarity': float(similarities[i][j])
                        })

                analysis_results.append({
                    'candidate': candidate,
                    'top_matches': top_matches
                })

            return analysis_results

        except Exception as e:
            logger.warning(f"Error in similarity analysis: {e}")
            return []

    def generate_threshold_report(self, results: List[Dict], output_file: str = None):
        """Generate comprehensive threshold analysis report."""
        print("\n" + "="*80)
        print("SIMILARITY THRESHOLD ANALYSIS REPORT")
        print("="*80)

        all_similarities = []
        correct_predictions = {0.5: 0, 0.6: 0, 0.7: 0, 0.75: 0, 0.8: 0, 0.85: 0, 0.9: 0}
        total_predictions = {thresh: 0 for thresh in correct_predictions.keys()}

        for result in results:
            print(f"\nSegment {result['segment_id']}: {result['text']}")
            print(f"Expected terms: {result['expected_terms']}")
            print("-" * 60)

            for analysis in result['similarity_analysis']:
                candidate = analysis['candidate']
                top_matches = analysis['top_matches']

                if top_matches:
                    best_match = top_matches[0]
                    similarity = best_match['similarity']
                    all_similarities.append(similarity)

                    print(f"Candidate: '{candidate}'")
                    print(f"  Top match: {best_match['canonical_term']} ({best_match['canonical_label']}) - {similarity:.3f}")

                    # Check if prediction would be correct for different thresholds
                    is_correct = best_match['canonical_term'] in result['expected_terms']

                    for thresh in correct_predictions.keys():
                        if similarity >= thresh:
                            total_predictions[thresh] += 1
                            if is_correct:
                                correct_predictions[thresh] += 1

                    # Show other top matches
                    for match in top_matches[1:3]:  # Show top 3
                        print(f"    {match['canonical_term']} ({match['canonical_label']}) - {match['similarity']:.3f}")
                    print()

        # Summary statistics
        print("\n" + "="*80)
        print("THRESHOLD PERFORMANCE ANALYSIS")
        print("="*80)

        if all_similarities:
            print(f"Similarity Score Statistics:")
            print(f"  Mean: {sum(all_similarities)/len(all_similarities):.3f}")
            print(f"  Min:  {min(all_similarities):.3f}")
            print(f"  Max:  {max(all_similarities):.3f}")
            print()

        print("Precision at Different Thresholds:")
        print("Threshold | Predictions | Correct | Precision")
        print("-" * 45)

        for thresh in sorted(correct_predictions.keys()):
            total = total_predictions[thresh]
            correct = correct_predictions[thresh]
            precision = correct / total if total > 0 else 0
            marker = " <-- Current" if thresh == SIMILARITY_THRESHOLD else ""
            print(f"   {thresh:.2f}   |     {total:3d}     |   {correct:3d}   |  {precision:.3f}{marker}")

        print("\nRecommendations:")
        print("- If precision is too low, increase threshold")
        print("- If too few predictions, decrease threshold")
        print("- Consider corpus-specific optimization")

        # Save CSV for further analysis
        if output_file:
            self._save_csv_analysis(results, output_file)
            print(f"\nDetailed analysis saved to: {output_file}")

    def _save_csv_analysis(self, results: List[Dict], output_file: str):
        """Save detailed analysis to CSV for further processing."""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['segment_id', 'text', 'expected_terms', 'candidate',
                         'best_canonical_term', 'best_similarity', 'is_correct']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                for analysis in result['similarity_analysis']:
                    candidate = analysis['candidate']
                    top_matches = analysis['top_matches']

                    if top_matches:
                        best_match = top_matches[0]
                        is_correct = best_match['canonical_term'] in result['expected_terms']

                        writer.writerow({
                            'segment_id': result['segment_id'],
                            'text': result['text'],
                            'expected_terms': '; '.join(result['expected_terms']),
                            'candidate': candidate,
                            'best_canonical_term': best_match['canonical_term'],
                            'best_similarity': best_match['similarity'],
                            'is_correct': is_correct
                        })


def create_sample_data():
    """Create sample economic segments for threshold tuning."""
    sample_segments = [
        {
            "text": "La inflación interanual alcanzó el 25.5 por ciento en diciembre.",
            "expected_terms": ["inflacion"]
        },
        {
            "text": "El producto bruto interno creció un 2.1% en el último trimestre.",
            "expected_terms": ["pbi"]
        },
        {
            "text": "Las reservas internacionales del banco central disminuyeron significativamente.",
            "expected_terms": ["reservas"]
        },
        {
            "text": "La tasa de desempleo se ubicó en 9.6 por ciento en el tercer trimestre.",
            "expected_terms": ["desempleo"]
        },
        {
            "text": "El dólar oficial cerró a $180 pesos en el mercado cambiario.",
            "expected_terms": ["dolar", "peso"]
        },
        {
            "text": "El déficit fiscal primario fue de 2.3% del PIB este año.",
            "expected_terms": ["deficit", "pbi"]
        },
        {
            "text": "La tasa de interés de política monetaria se mantuvo en 75%.",
            "expected_terms": ["tasa_interes"]
        },
        {
            "text": "La oferta monetaria M2 creció un 45% en términos interanuales.",
            "expected_terms": ["m2"]
        },
        {
            "text": "Las importaciones cayeron un 15% respecto al año anterior.",
            "expected_terms": ["importaciones"]
        },
        {
            "text": "El gobierno redujo los subsidios a la energía en un 30%.",
            "expected_terms": ["subsidios"]
        },
        {
            "text": "Los precios al consumidor subieron considerablemente este mes.",
            "expected_terms": ["inflacion"]
        },
        {
            "text": "La actividad económica mostró signos de recuperación gradual.",
            "expected_terms": ["pbi"]
        }
    ]

    return sample_segments


def main():
    """Main function for threshold tuning."""
    parser = argparse.ArgumentParser(description='Tune similarity threshold for economic term detection')
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD,
                       help=f'Current similarity threshold (default: {SIMILARITY_THRESHOLD})')
    parser.add_argument('--samples', type=str, default=None,
                       help='JSON file with sample segments (default: use built-in samples)')
    parser.add_argument('--output', type=str, default='threshold_analysis.csv',
                       help='Output CSV file for detailed analysis')

    args = parser.parse_args()

    print(f"Threshold Tuning for Economic Term Detection")
    print(f"Current threshold: {args.threshold}")
    print("="*60)

    # Initialize tuner
    try:
        tuner = ThresholdTuner()
    except Exception as e:
        logger.error(f"Failed to initialize tuner: {e}")
        sys.exit(1)

    # Load sample segments
    if args.samples and args.samples != 'sample_segments.json':
        try:
            with open(args.samples, 'r', encoding='utf-8') as f:
                sample_segments = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sample file {args.samples}: {e}")
            sys.exit(1)
    else:
        sample_segments = create_sample_data()
        # Save default samples for reference
        with open('sample_segments.json', 'w', encoding='utf-8') as f:
            json.dump(sample_segments, f, indent=2, ensure_ascii=False)
        print(f"Created sample_segments.json with {len(sample_segments)} test segments")

    print(f"Analyzing {len(sample_segments)} sample segments...")

    # Analyze segments
    results = tuner.analyze_sample_segments(sample_segments)

    # Generate report
    tuner.generate_threshold_report(results, args.output)

    print(f"\nThreshold tuning analysis complete!")
    print(f"Current threshold: {SIMILARITY_THRESHOLD}")
    print(f"To change threshold, edit SIMILARITY_THRESHOLD in detect_economic_terms_with_embeddings.py")


if __name__ == '__main__':
    main()