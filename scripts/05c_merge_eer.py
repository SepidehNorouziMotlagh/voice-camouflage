"""
Merge ECAPA and WavLM EER results into one table.
"""
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "eer_results"

ecapa = pd.read_csv(OUTPUT_DIR / "eer_ecapa.csv")
wavlm = pd.read_csv(OUTPUT_DIR / "eer_wavlm.csv")

merged = pd.merge(ecapa, wavlm, on="condition")
merged.to_csv(OUTPUT_DIR / "eer_results.csv", index=False)

print(f"✅ Merged EER results saved to {OUTPUT_DIR / 'eer_results.csv'}")
print(merged.to_string(index=False))