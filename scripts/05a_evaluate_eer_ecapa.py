"""
Evaluate EER Using ECAPA-TDNN

This script computes Equal Error Rate (EER) for all mixture conditions using
the ECAPA-TDNN speaker verification model. It constructs genuine trials
(target enrollment vs target mixture) and impostor trials (other-speaker
enrollment vs target mixture) for each condition and crowd size.

Uses only SpeechBrain (no Transformers) to avoid library conflicts.
"""
import json
import random
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from sklearn.metrics import roc_curve
from tqdm import tqdm
import pandas as pd

# ---------- Fixed random seed for reproducibility ----------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# ----------------------------------------------------------

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
MANIFEST_FILE = BASE_DIR / "config" / "utterance_roles.json"
MIXTURES_DIR = BASE_DIR / "data" / "mixtures"
OUTPUT_DIR = BASE_DIR / "results" / "eer_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
CONDITIONS = ["random_rep1", "random_rep2", "random_rep3", "ecapa_only", "wavlm_only", "ensemble"]
CROWD_SIZES = [1, 2, 4, 6, 8]
# ----------------------------------------------------------------

def compute_eer(scores_genuine, scores_impostor):
    """
    Compute Equal Error Rate from genuine and impostor scores.

    Args:
        scores_genuine: Similarity scores for genuine trials
        scores_impostor: Similarity scores for impostor trials

    Returns:
        float: Equal Error Rate
    """
    labels = np.concatenate([np.ones_like(scores_genuine), np.zeros_like(scores_impostor)])
    scores = np.concatenate([scores_genuine, scores_impostor])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    return (fpr[idx] + fnr[idx]) / 2

def get_embedding(model, audio_path):
    """
    Extract ECAPA-TDNN embedding from an audio file.

    Args:
        model: Loaded ECAPA-TDNN model
        audio_path: Path to audio file

    Returns:
        np.ndarray: Speaker embedding
    """
    signal, fs = sf.read(audio_path)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if fs != SR:
        import librosa
        signal = librosa.resample(signal.astype(np.float32), orig_sr=fs, target_sr=SR)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(signal_tensor)
    return emb.squeeze().cpu().numpy()

def cosine_sim(a, b):
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        float: Cosine similarity
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def main():
    print("=" * 60)
    print("EER EVALUATION (ECAPA-TDNN)")
    print("=" * 60)

    # Load the utterance role manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    targets = manifest["targets"]

    print("Loading ECAPA-TDNN model...")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )

    results = []

    for cond in CONDITIONS:
        print(f"\nProcessing condition: {cond}")
        cond_dir = MIXTURES_DIR / cond
        speakers = list(targets.keys())

        genuine_all = []
        impostor_all = []

        for spk in tqdm(speakers, desc="Speaker loop"):
            # Get enrollment utterance embedding
            enroll_path = targets[spk]["enrollment"][0]
            enroll_emb = get_embedding(model, enroll_path)

            # For each crowd size, evaluate mixtures
            for k in CROWD_SIZES:
                mix_path = cond_dir / f"{spk}_top_{k}.wav"
                if not mix_path.exists():
                    continue

                # Genuine trial: same speaker
                mix_emb = get_embedding(model, mix_path)
                genuine_sim = cosine_sim(enroll_emb, mix_emb)
                genuine_all.append(genuine_sim)

                # Impostor trial: random different speaker
                imp_candidates = [s for s in speakers if s != spk]
                imp_spk = np.random.choice(imp_candidates)
                imp_enroll_path = targets[imp_spk]["enrollment"][0]
                imp_enroll_emb = get_embedding(model, imp_enroll_path)
                imp_sim = cosine_sim(imp_enroll_emb, mix_emb)
                impostor_all.append(imp_sim)

        # Compute EER for this condition
        eer = compute_eer(np.array(genuine_all), np.array(impostor_all))
        results.append({"condition": cond, "eer_ecapa": eer})
        print(f"  EER (ECAPA): {eer:.4f}")

    # Save results
    df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "eer_ecapa.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Saved ECAPA EER results to {output_file}")

if __name__ == "__main__":
    main()