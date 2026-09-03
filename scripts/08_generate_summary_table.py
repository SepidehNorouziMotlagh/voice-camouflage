"""
Generate Final Results Summary Table

This script merges the EER, WER, and STOI results into a single summary table
for the paper. It extracts WER and STOI at four interfering speakers (the
"sweet spot" where the strongest selection effect was observed) and combines
them with the EER results for all conditions.

The output is a clean table containing:
    condition, EER (ECAPA), EER (WavLM), WER, STOI
"""
import pandas as pd
from pathlib import Path

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "results"

EER_FILE = OUTPUT_DIR / "eer_results" / "eer_results.csv"
WER_FILE = OUTPUT_DIR / "wer_results" / "wer_aggregate_merged.csv"
STOI_FILE = OUTPUT_DIR / "stoi_results" / "stoi_aggregate.csv"
OUTPUT_FILE = OUTPUT_DIR / "final_results_table.csv"
# ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("GENERATE FINAL RESULTS SUMMARY TABLE")
    print("=" * 60)

    # Check that all input files exist
    for file_path in [EER_FILE, WER_FILE, STOI_FILE]:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            print("   Please run the evaluation scripts first.")
            return

    # Load EER results (all conditions)
    print(f"Loading EER from {EER_FILE}")
    eer = pd.read_csv(EER_FILE)

    # Load WER results (filter to 4 talkers)
    print(f"Loading WER from {WER_FILE}")
    wer = pd.read_csv(WER_FILE)
    wer_4 = wer[wer["crowd_size"] == 4]

    # Load STOI results (filter to 4 talkers)
    print(f"Loading STOI from {STOI_FILE}")
    stoi = pd.read_csv(STOI_FILE)
    stoi_4 = stoi[stoi["crowd_size"] == 4]

    # Merge all results on condition name
    print("Merging results...")
    merged = pd.merge(eer, wer_4, on="condition")
    merged = pd.merge(merged, stoi_4, on="condition")

    # Keep only the relevant columns and rename for clarity
    merged = merged[["condition", "eer_ecapa", "eer_wavlm", "mean_x", "mean_y"]]
    merged.columns = ["Condition", "EER (ECAPA)", "EER (WavLM)", "WER", "STOI"]

    # Save the final table
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Final results table saved to {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print("FINAL RESULTS TABLE (4 Interfering Speakers)")
    print("=" * 60)
    print(merged.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    main()