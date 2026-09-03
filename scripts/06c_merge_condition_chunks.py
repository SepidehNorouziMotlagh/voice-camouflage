"""
Merge Chunks for a Single Condition

This script merges all chunk CSV files for a specific condition into a single
CSV file containing WER results for all 100 speakers.

Usage:
    python 06c_merge_condition_chunks.py <condition>

Example:
    python 06c_merge_condition_chunks.py random_rep1

Available conditions:
    random_rep1, random_rep2, random_rep3, ecapa_only, wavlm_only, ensemble
"""
import pandas as pd
from pathlib import Path
import sys

# ------------------------- Configuration -------------------------
# IMPORTANT: This script is designed to run on Google Colab.
# Update BASE_DIR to match your Google Drive path.
BASE_DIR = Path("/content/drive/MyDrive/voice_camouflage")
OUTPUT_DIR = BASE_DIR / "results" / "wer_results"
# ----------------------------------------------------------------

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python 06c_merge_condition_chunks.py <condition>")
        print("Available: random_rep1, random_rep2, random_rep3, ecapa_only, wavlm_only, ensemble")
        sys.exit(1)

    condition = sys.argv[1]

    # Find all chunk files for this condition
    chunk_files = sorted(OUTPUT_DIR.glob(f"wer_{condition}_chunk*.csv"))
    if not chunk_files:
        print(f"No chunk files found for {condition}")
        print("Please run 06b_wer_condition_chunked.py for all chunks first.")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files for {condition}:")
    for f in chunk_files:
        print(f"  {f.name}")

    # Load and concatenate all chunk files
    dfs = []
    for f in chunk_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"Loaded {f.name}: {len(df)} rows")

    merged = pd.concat(dfs, ignore_index=True)

    # Save merged results
    output_file = OUTPUT_DIR / f"wer_{condition}.csv"
    merged.to_csv(output_file, index=False)

    print(f"\n✅ Merged {condition} saved to {output_file}")
    print(f"   Total rows: {len(merged)}")

    # Print aggregate statistics
    agg = merged.groupby(["crowd_size"])["WER"].agg(["mean", "std", "count"]).reset_index()
    print("\nAggregate (merged):")
    print(agg.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    main()