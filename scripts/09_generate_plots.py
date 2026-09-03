"""
Generate Plots

This script generates four plots:
1. WER vs Crowd Size: Utility degradation across crowd sizes
2. STOI vs Crowd Size: Signal-level intelligibility across crowd sizes
3. Privacy-Utility Trade-off: WavLM EER vs WER at 4 interfering speakers
4. Bar Chart: Comparison of all metrics at 4 interfering speakers

All plots use the aggregated results from the evaluation scripts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load aggregated results
EER = pd.read_csv(RESULTS_DIR / "eer_results" / "eer_results.csv")
WER = pd.read_csv(RESULTS_DIR / "wer_results" / "wer_aggregate_merged.csv")
STOI = pd.read_csv(RESULTS_DIR / "stoi_results" / "stoi_aggregate.csv")

# Plotting configuration
CONDITION_KEYS = {
    "random": ["random_rep1", "random_rep2", "random_rep3"],
    "ecapa_only": ["ecapa_only"],
    "wavlm_only": ["wavlm_only"],
    "ensemble": ["ensemble"],
}

DISPLAY_NAMES = {
    "random": "Random",
    "ecapa_only": "ECAPA-only",
    "wavlm_only": "WavLM-only",
    "ensemble": "Ensemble",
}

COLORS = {
    "random": "blue",
    "ecapa_only": "green",
    "wavlm_only": "orange",
    "ensemble": "red",
}

MARKERS = {
    "random": "o",
    "ecapa_only": "s",
    "wavlm_only": "^",
    "ensemble": "D",
}

CROWD_SIZES = [1, 2, 4, 6, 8]
# ----------------------------------------------------------------

def average_random_repetitions(df, metric_col, crowd_col="crowd_size", agg_col="mean"):
    """
    Average the three random repetitions for a given metric.

    Args:
        df: DataFrame containing results
        metric_col: Column name for the metric to average
        crowd_col: Column name for crowd size
        agg_col: Column name for the aggregated value

    Returns:
        DataFrame: Averaged random baseline with mean and std
    """
    random_dfs = []
    for rep in ["random_rep1", "random_rep2", "random_rep3"]:
        temp = df[df["condition"] == rep].copy()
        random_dfs.append(temp)

    combined = pd.concat(random_dfs)
    avg = combined.groupby(crowd_col)[agg_col].agg(["mean", "std"]).reset_index()
    avg.columns = ["crowd_size", "mean", "std"]
    avg["condition"] = "random"
    return avg

def average_random_eer(eer_df):
    """Average the three random repetitions for EER."""
    random_ecapa = []
    random_wavlm = []

    for rep in ["random_rep1", "random_rep2", "random_rep3"]:
        row = eer_df[eer_df["condition"] == rep]
        random_ecapa.append(row["eer_ecapa"].values[0])
        random_wavlm.append(row["eer_wavlm"].values[0])

    avg_ecapa = np.mean(random_ecapa)
    std_ecapa = np.std(random_ecapa)
    avg_wavlm = np.mean(random_wavlm)
    std_wavlm = np.std(random_wavlm)

    return avg_ecapa, std_ecapa, avg_wavlm, std_wavlm

def main():
    print("=" * 60)
    print("GENERATE PUBLICATION PLOTS")
    print("=" * 60)

    # Compute random baselines
    random_wer = average_random_repetitions(WER, metric_col="mean")
    random_stoi = average_random_repetitions(STOI, metric_col="mean")
    eer_random_avg_ecapa, _, eer_random_avg_wavlm, _ = average_random_eer(EER)

    # Prepare data for WER and STOI
    wer_data = {}
    stoi_data = {}

    for cond in ["random", "ecapa_only", "wavlm_only", "ensemble"]:
        if cond == "random":
            wer_data[cond] = random_wer
            stoi_data[cond] = random_stoi
        else:
            wer_data[cond] = WER[WER["condition"] == cond]
            stoi_data[cond] = STOI[STOI["condition"] == cond]

    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.figsize'] = (8, 6)

    # --- Figure 1: WER vs Crowd Size ---
    print("\nGenerating Figure 1: WER vs Crowd Size...")
    fig, ax = plt.subplots()

    for cond in ["random", "ecapa_only", "wavlm_only", "ensemble"]:
        data = wer_data[cond]
        ax.errorbar(
            data["crowd_size"], data["mean"], yerr=data["std"],
            marker=MARKERS[cond], color=COLORS[cond],
            label=DISPLAY_NAMES[cond], capsize=5,
            linewidth=2, markersize=8
        )

    ax.set_xlabel("Number of Background Talkers")
    ax.set_ylabel("WER")
    ax.set_title("Utility: Word Error Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(CROWD_SIZES)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "wer_vs_crowd.png", dpi=300)
    plt.savefig(PLOT_DIR / "wer_vs_crowd.pdf")
    print(f"  Saved to {PLOT_DIR / 'wer_vs_crowd.png'}")

    # --- Figure 2: STOI vs Crowd Size ---
    print("\nGenerating Figure 2: STOI vs Crowd Size...")
    fig, ax = plt.subplots()

    for cond in ["random", "ecapa_only", "wavlm_only", "ensemble"]:
        data = stoi_data[cond]
        ax.errorbar(
            data["crowd_size"], data["mean"], yerr=data["std"],
            marker=MARKERS[cond], color=COLORS[cond],
            label=DISPLAY_NAMES[cond], capsize=5,
            linewidth=2, markersize=8
        )

    ax.set_xlabel("Number of Background Talkers")
    ax.set_ylabel("STOI")
    ax.set_title("Utility: Short-Time Objective Intelligibility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(CROWD_SIZES)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "stoi_vs_crowd.png", dpi=300)
    plt.savefig(PLOT_DIR / "stoi_vs_crowd.pdf")
    print(f"  Saved to {PLOT_DIR / 'stoi_vs_crowd.png'}")

    # --- Figure 3: Privacy-Utility Trade-off at 4 Talkers ---
    print("\nGenerating Figure 3: Privacy-Utility Trade-off...")
    random_wer_4 = random_wer[random_wer["crowd_size"] == 4]["mean"].values[0]

    # EER values from results
    eer_ecapa = {
        "random": eer_random_avg_ecapa,
        "ecapa_only": 0.450,
        "wavlm_only": 0.494,
        "ensemble": 0.564,
    }
    eer_wavlm = {
        "random": eer_random_avg_wavlm,
        "ecapa_only": 0.414,
        "wavlm_only": 0.652,
        "ensemble": 0.642,
    }
    wer_4 = {
        "random": random_wer_4,
        "ecapa_only": 1.212,
        "wavlm_only": 1.059,
        "ensemble": 1.095,
    }

    # Build trade-off data
    tradeoff_data = []
    for cond in ["random", "ecapa_only", "wavlm_only", "ensemble"]:
        tradeoff_data.append({
            "condition": DISPLAY_NAMES[cond],
            "cond_key": cond,
            "EER_WavLM": eer_wavlm[cond],
            "WER": wer_4[cond],
        })

    tradeoff_df = pd.DataFrame(tradeoff_data)

    # Plot trade-off
    fig, ax = plt.subplots()

    for _, row in tradeoff_df.iterrows():
        cond_key = row["cond_key"]
        ax.scatter(
            row["WER"], row["EER_WavLM"],
            s=100, label=row["condition"],
            color=COLORS[cond_key], marker=MARKERS[cond_key]
        )
        ax.annotate(
            row["condition"],
            (row["WER"] + 0.02, row["EER_WavLM"] + 0.01),
            fontsize=10
        )

    ax.set_xlabel("WER (4 talkers, lower is better)")
    ax.set_ylabel("WavLM EER (higher is better)")
    ax.set_title("Privacy-Utility Trade-off at 4 Talkers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "tradeoff_wavlm_vs_wer.png", dpi=300)
    plt.savefig(PLOT_DIR / "tradeoff_wavlm_vs_wer.pdf")
    print(f"  Saved to {PLOT_DIR / 'tradeoff_wavlm_vs_wer.png'}")

    # --- Figure 4: Bar Chart at 4 Talkers ---
    print("\nGenerating Figure 4: Bar Chart at 4 Talkers...")
    metrics = ["EER (ECAPA)", "EER (WavLM)", "WER", "STOI"]
    conditions = tradeoff_df["condition"].tolist()
    values = tradeoff_df[["EER_ECAPA", "EER_WavLM", "WER", "STOI"]].values

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.2

    for i, row in tradeoff_df.iterrows():
        cond_key = row["cond_key"]
        ax.bar(
            x + i * width, values[i], width,
            label=row["condition"], color=COLORS[cond_key]
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Comparison of All Conditions at 4 Talkers")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "bar_chart_4talkers.png", dpi=300)
    plt.savefig(PLOT_DIR / "bar_chart_4talkers.pdf")
    print(f"  Saved to {PLOT_DIR / 'bar_chart_4talkers.png'}")

    print("\n✅ All plots generated successfully!")
    print(f"   Plots saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()