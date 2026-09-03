"""
Generate Selection Lists for All Strategies

This script creates the background speaker selection lists for four strategies:
- random: random selection (3 repetitions per target per crowd size)
- ecapa_only: lowest ECAPA similarity
- wavlm_only: lowest WavLM similarity
- ensemble: lowest average of normalised ranks from both models

The selection lists are saved as JSON files for use in mixture generation.
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
SCORES_FILE = BASE_DIR / "results" / "embeddings" / "all_scores.csv"
OUTPUT_DIR = BASE_DIR / "data" / "selections"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CROWD_SIZES = [1, 2, 4, 6, 8]
RANDOM_SEED = 42
RANDOM_REPETITIONS = 3

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("GENERATE SELECTION LISTS")
    print("=" * 60)

    # Load similarity scores
    df = pd.read_csv(SCORES_FILE)
    # Convert speaker IDs to strings to avoid JSON serialisation issues
    df["target"] = df["target"].astype(str)
    df["background"] = df["background"].astype(str)

    targets = df["target"].unique()
    backgrounds = df["background"].unique()

    print(f"Targets: {len(targets)}")
    print(f"Backgrounds: {len(backgrounds)}")
    print(f"Pairs: {len(df)}")

    selections = {
        "random": {},
        "ecapa_only": {},
        "wavlm_only": {},
        "ensemble": {},
    }

    for target in targets:
        target_df = df[df["target"] == target].copy()

        # --- Random selection (3 repetitions per crowd size) ---
        random_selections = {}
        for rep in range(RANDOM_REPETITIONS):
            random.seed(RANDOM_SEED + rep)
            random_selections[f"rep_{rep+1}"] = {}
            for k in CROWD_SIZES:
                selected = random.sample(list(backgrounds), k)
                random_selections[f"rep_{rep+1}"][f"top_{k}"] = selected
        selections["random"][target] = random_selections

        # --- ECAPA-only selection ---
        ecapa_sorted = target_df.sort_values("ecapa_similarity", ascending=True)  # lowest first
        ecapa_selected = {}
        for k in CROWD_SIZES:
            ecapa_selected[f"top_{k}"] = ecapa_sorted.head(k)["background"].tolist()
        selections["ecapa_only"][target] = ecapa_selected

        # --- WavLM-only selection ---
        wavlm_sorted = target_df.sort_values("wavlm_similarity", ascending=True)  # lowest first
        wavlm_selected = {}
        for k in CROWD_SIZES:
            wavlm_selected[f"top_{k}"] = wavlm_sorted.head(k)["background"].tolist()
        selections["wavlm_only"][target] = wavlm_selected

        # --- Ensemble selection (rank fusion) ---
        # 1. Rank by ECAPA (ascending, lowest = most dissimilar)
        target_df["ecapa_rank"] = target_df["ecapa_similarity"].rank(method="dense", ascending=True)
        # 2. Rank by WavLM (ascending, lowest = most dissimilar)
        target_df["wavlm_rank"] = target_df["wavlm_similarity"].rank(method="dense", ascending=True)
        # 3. Normalise ranks to [0, 1]
        target_df["ecapa_rank_norm"] = (target_df["ecapa_rank"] - 1) / (len(backgrounds) - 1)
        target_df["wavlm_rank_norm"] = (target_df["wavlm_rank"] - 1) / (len(backgrounds) - 1)
        # 4. Ensemble score = average of normalised ranks
        target_df["ensemble_score"] = (target_df["ecapa_rank_norm"] + target_df["wavlm_rank_norm"]) / 2
        # 5. Sort by ensemble score (lowest = most dissimilar under both models)
        ensemble_sorted = target_df.sort_values("ensemble_score", ascending=True)
        ensemble_selected = {}
        for k in CROWD_SIZES:
            ensemble_selected[f"top_{k}"] = ensemble_sorted.head(k)["background"].tolist()
        selections["ensemble"][target] = ensemble_selected

    # --- Save each selection condition ---
    for cond_name, cond_data in selections.items():
        output_path = OUTPUT_DIR / f"{cond_name}_backgrounds.json"
        with open(output_path, "w") as f:
            json.dump(cond_data, f, indent=2)
        print(f"✅ Saved {cond_name} selections to {output_path}")

    print("\n" + "=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)
    print(f"Random: {RANDOM_REPETITIONS} repetitions, {len(CROWD_SIZES)} crowd sizes")
    print(f"ECAPA-only: {len(CROWD_SIZES)} crowd sizes")
    print(f"WavLM-only: {len(CROWD_SIZES)} crowd sizes")
    print(f"Ensemble: {len(CROWD_SIZES)} crowd sizes")

if __name__ == "__main__":
    main()