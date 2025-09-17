#!/usr/bin/env python3
"""
Economic Term Detection with Embeddings

Advanced NLP pipeline for detecting economic indicators and technical terms in Spanish transcriptions.
Uses spaCy + PhraseMatcher for exact matches and SBERT + FAISS for semantic similarity.

Author: AI Engineer Senior
Date: 2025-01-17
"""

import sys
import os
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration and detector
from config.config_loader import get_config
from detectors.economic_detector import EconomicTermDetector


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

    # Load configuration
    try:
        config = get_config()
        if not config.validate():
            logger.error("Configuration validation failed")
            sys.exit(1)
        logger.info("Configuration loaded and validated successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        print(f"Error: Failed to load configuration: {e}")
        sys.exit(1)

    # Create output directories
    output_dirs = config.get_output_dirs()
    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)

    # Extract base filename for output files
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    print(f"Processing {input_file}...")
    print(f"Configuration:")
    print(f"  - Embeddings enabled: {config.use_embeddings}")
    print(f"  - Similarity threshold: {config.similarity_threshold}")
    print(f"  - Context window: {config.context_window} characters")
    print()

    # Initialize detector
    try:
        detector = EconomicTermDetector(config)
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
        print(f"  - JSON: {output_dirs['analysis']}/{base_filename}_detected_terms.json")
        print(f"  - Markdown: {output_dirs['analysis']}/{base_filename}_detected_terms.md")

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        print(f"Error: Failed to process file: {e}")
        sys.exit(1)

    print("\nEconomic term detection completed successfully!")


if __name__ == "__main__":
    main()