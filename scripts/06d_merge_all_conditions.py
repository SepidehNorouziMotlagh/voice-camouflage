"""
Merge All WER Conditions

This script merges all condition CSV files into a single file containing
WER results for all conditions, speakers, and crowd sizes.

It also generates an aggregate summary table and a pivot table for easier
visualisation of mean WER across conditions and crowd sizes.
"""
import pandas as pd
from pathlib import Path
import sys

# ------------------------- Configuration -------------------------
# IMPORTANT: This script is designed to run on Google Colab.
# Update BASE_DIR to match your Google Drive path.
BASE_DIR = Path("/content/drive/MyDrive/voice_camouflage")
OUTPUT_DIR = BASE_DIR / "results" / "wer_results"

CONDITIONS = ["random_rep1", "random_rep2", "random_rep3", "ecapa_only", "wavlm_only", "ensemble"]
# ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("MERGE ALL WER CONDITIONS")
    print("=" * 60)

    # Check if output directory exists
    if not OUTPUT_DIR.exists():
        print(f"❌ Output directory not found: {OUTPUT_DIR}")
        print("   Please run 06b_wer_condition_chunked.py and 06c_merge_condition_chunks.py first.")
        sys.exit(1)

    # Load all condition files
    dfs = []
    for cond in CONDITIONS:
        f = OUTPUT_DIR / f"wer_{cond}.csv"
        if f.exists():
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"Loaded {cond}: {len(df)} rows")
        else:
            print(f"⚠️ {cond} not found (run 06c_merge_condition_chunks.py first)")

    if not dfs:
        print("\nNo files to merge. Run 06c_merge_condition_chunks.py for each condition first.")
        sys.exit(1)

    # Merge all condition files
    merged = pd.concat(dfs, ignore_index=True)

    # Save merged results
    output_file = OUTPUT_DIR / "wer_merged.csv"
    merged.to_csv(output_file, index=False)

    print(f"\n✅ Merged WER results saved to {output_file}")
    print(f"   Total rows: {len(merged)}")

    # Aggregate by condition and crowd size
    agg = merged.groupby(["condition", "crowd_size"])["WER"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["condition", "crowd_size", "mean", "std", "count"]

    # Save aggregate results
    agg_file = OUTPUT_DIR / "wer_aggregate_merged.csv"
    agg.to_csv(agg_file, index=False)

    print("\nAggregate (all conditions):")
    print(agg.to_string(index=False, float_format="%.3f"))

    # Pivot for easier reading
    pivot = agg.pivot(index="condition", columns="crowd_size", values="mean")
    print("\nPivot Table (mean WER):")
    print(pivot.round(3))

if __name__ == "__main__":
    main()