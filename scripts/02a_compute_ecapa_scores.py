"""
Compute ECAPA Similarities for All Target-Background Pairs

This script loads the utterance role manifest, computes ECAPA-TDNN embeddings
for all target and background speakers, and calculates cosine similarity
for every target-background pair.

The output is a CSV file containing all pairwise similarities, which is used
to generate selection lists for different strategies.

Uses only SpeechBrain (no transformers) to avoid library conflicts.
"""
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from tqdm import tqdm
import pandas as pd

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
MANIFEST_FILE = BASE_DIR / "config" / "utterance_roles.json"
OUTPUT_DIR = BASE_DIR / "results" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
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
    signal, fs = sf.read(path)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if fs != sr:
        import librosa
        signal = librosa.resample(signal.astype(np.float32), orig_sr=fs, target_sr=sr)
    return signal.astype(np.float32)

def ecapa_embedding(model, audio_path):
    """
    Extract ECAPA-TDNN embedding from an audio file.

    Args:
        model: Loaded ECAPA-TDNN model
        audio_path: Path to audio file

    Returns:
        np.ndarray: Speaker embedding
    """
    signal = load_audio_mono(audio_path)
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
    print("ECAPA SIMILARITY COMPUTATION")
    print("=" * 60)

    # Load the utterance role manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    targets = manifest["targets"]
    backgrounds = manifest["backgrounds"]

    print(f"Loading ECAPA-TDNN model...")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    # Compute target embeddings (average of selection set utterances)
    target_emb = {}
    for spk, roles in tqdm(targets.items(), desc="Target embeddings"):
        sel_files = roles["selection"]
        if not sel_files:
            continue
        embs = [ecapa_embedding(model, f) for f in sel_files]
        target_emb[spk] = np.mean(embs, axis=0)

    # Compute background embeddings (first utterance per speaker)
    bg_emb = {}
    for bg_spk, bg_files in tqdm(backgrounds.items(), desc="Background embeddings"):
        if not bg_files:
            continue
        bg_emb[bg_spk] = ecapa_embedding(model, bg_files[0])

    # Compute similarity matrix for all target-background pairs
    rows = []
    for target_spk, t_emb in tqdm(target_emb.items(), desc="Computing similarities"):
        for bg_spk, b_emb in bg_emb.items():
            sim = cosine_sim(t_emb, b_emb)
            rows.append({
                "target": target_spk,
                "background": bg_spk,
                "ecapa_similarity": sim
            })

    # Save results
    df = pd.DataFrame(rows)
    output_file = OUTPUT_DIR / "ecapa_all_scores.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Saved ECAPA scores to {output_file}")
    print(f"   {len(df)} rows ({len(targets)} targets × {len(backgrounds)} backgrounds)")

if __name__ == "__main__":
    main()