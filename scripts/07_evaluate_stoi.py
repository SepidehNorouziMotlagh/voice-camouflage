"""
Evaluate STOI for All Conditions

This script computes Short-Time Objective Intelligibility (STOI) for all
mixture conditions. STOI is a signal-level intelligibility metric that
measures the correlation between the clean reference and the processed signal.

The script runs locally on CPU and produces two output files:
    - stoi_results.csv: Per-speaker, per-condition STOI scores
    - stoi_aggregate.csv: Mean, std, and count per condition and crowd size
"""
import json
import numpy as np
import pandas as pd
from pystoi import stoi
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
MANIFEST_FILE = BASE_DIR / "config" / "utterance_roles.json"
MIXTURES_DIR = BASE_DIR / "data" / "mixtures"
OUTPUT_DIR = BASE_DIR / "results" / "stoi_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
CONDITIONS = [
    "random_rep1", "random_rep2", "random_rep3",
    "ecapa_only", "wavlm_only", "ensemble"
]
CROWD_SIZES = [1, 2, 4, 6, 8]
# ----------------------------------------------------------------

def load_audio_mono(path, sr=SR):
    """
    Load audio file, convert to mono, and resample to target sample rate.

    Args:
        path: Path to audio file
        sr: Target sample rate

    Returns:
        np.ndarray: Mono audio signal at target sample rate
    """
    y, fs = sf.read(path)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if fs != sr:
        import librosa
        y = librosa.resample(y.astype(np.float32), orig_sr=fs, target_sr=sr)
    return y.astype(np.float32)

def main():
    print("=" * 60)
    print("STOI EVALUATION")
    print("=" * 60)

    # Load the utterance role manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    targets = manifest["targets"]

    results = []

    for spk in tqdm(targets, desc="Processing speakers"):
        # Load clean reference
        clean_path = targets[spk]["evaluation"][0]
        clean = load_audio_mono(clean_path)

        # For each condition and crowd size
        for cond in CONDITIONS:
            cond_dir = MIXTURES_DIR / cond

            for k in CROWD_SIZES:
                mix_path = cond_dir / f"{spk}_top_{k}.wav"
                if not mix_path.exists():
                    continue

                mix = load_audio_mono(mix_path)

                # Trim to same length (STOI requires equal length signals)
                min_len = min(len(clean), len(mix))
                clean_trim = clean[:min_len]
                mix_trim = mix[:min_len]

                # Compute STOI
                try:
                    stoi_score = stoi(clean_trim, mix_trim, SR, extended=False)
                except Exception as e:
                    print(f"STOI failed for {mix_path}: {e}")
                    stoi_score = np.nan

                results.append({
                    "speaker": spk,
                    "condition": cond,
                    "crowd_size": k,
                    "STOI": stoi_score
                })

    # Save per-speaker results
    df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "stoi_results.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Saved STOI results to {output_file}")

    # Aggregate by condition and crowd size
    agg = df.groupby(["condition", "crowd_size"])["STOI"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["condition", "crowd_size", "mean", "std", "count"]

    # Save aggregate results
    agg_file = OUTPUT_DIR / "stoi_aggregate.csv"
    agg.to_csv(agg_file, index=False)

    print("\n" + "=" * 60)
    print("STOI AGGREGATE RESULTS")
    print("=" * 60)
    print(agg.to_string(index=False, float_format="%.3f"))

    # Pivot for easier reading
    pivot = agg.pivot(index="condition", columns="crowd_size", values="mean")
    print("\nPivot Table (mean STOI):")
    print(pivot.round(3))

if __name__ == "__main__":
    main()