"""
Merge ECAPA and WavLM Similarity Scores

This script merges the ECAPA and WavLM similarity scores computed in the
previous steps into a single CSV file. The merged file contains all pairwise
similarities for every target-background pair under both models.

This combined file is used as input for generating selection lists.
"""
import pandas as pd
from pathlib import Path

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "results" / "embeddings"
OUTPUT_DIR = BASE_DIR / "results" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ECAPA_FILE = INPUT_DIR / "ecapa_all_scores.csv"
WAVLM_FILE = INPUT_DIR / "wavlm_all_scores.csv"
OUTPUT_FILE = OUTPUT_DIR / "all_scores.csv"
# ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("MERGE ECAPA AND WAVLM SCORES")
    print("=" * 60)

    # Check that both input files exist
    if not ECAPA_FILE.exists():
        print(f"❌ ECAPA file not found: {ECAPA_FILE}")
        print("   Please run 02a_compute_ecapa_scores.py first.")
        return

    if not WAVLM_FILE.exists():
        print(f"❌ WavLM file not found: {WAVLM_FILE}")
        print("   Please run 02b_compute_wavlm_scores.py first.")
        return

    # Load the scores
    print(f"Loading ECAPA scores from {ECAPA_FILE}")
    ecapa = pd.read_csv(ECAPA_FILE)

    print(f"Loading WavLM scores from {WAVLM_FILE}")
    wavlm = pd.read_csv(WAVLM_FILE)

    # Merge on target and background speaker IDs
    print("Merging scores...")
    merged = pd.merge(ecapa, wavlm, on=["target", "background"])

    # Save merged results
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Merged scores saved to {OUTPUT_FILE}")
    print(f"   Total rows: {len(merged)}")
    print(f"   Columns: {list(merged.columns)}")

if __name__ == "__main__":
    main()