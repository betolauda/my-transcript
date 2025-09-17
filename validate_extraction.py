#!/usr/bin/env python3
"""Quick validation of DetectedTerm extraction"""

import sys

def test_detectedterm_extraction():
    """Test that DetectedTerm can be imported from both locations."""
    print("Testing DetectedTerm extraction...")

    try:
        # Test import from new location
        from models.detected_term import DetectedTerm as NewDetectedTerm
        print("✓ DetectedTerm successfully imported from models/detected_term")

        # Test import from main module (should use the imported one)
        from detect_economic_terms_with_embeddings import DetectedTerm as MainDetectedTerm
        print("✓ DetectedTerm successfully imported from main module")

        # Test they are the same class
        if NewDetectedTerm == MainDetectedTerm:
            print("✓ Both imports reference the same class")
        else:
            print("❌ Different classes - import issue")
            return False

        # Test creating an instance
        term = NewDetectedTerm(
            snippet="test",
            timestamp=1.0,
            canonical_term="test",
            matched_text="test",
            numeric_value=None,
            numeric_text=None,
            confidence=1.0,
            match_type="exact",
            context="test context"
        )

        print("✓ DetectedTerm instance created successfully")
        print(f"  - snippet: {term.snippet}")
        print(f"  - group_id: '{term.group_id}' (default)")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_detectedterm_extraction()
    sys.exit(0 if success else 1)