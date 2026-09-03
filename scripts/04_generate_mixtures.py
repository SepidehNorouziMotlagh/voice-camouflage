"""
Generate Voice Camouflage Mixtures

This script generates audio mixtures for all selection conditions using the
precomputed background speaker selection lists:
- random (3 repetitions per target)
- ecapa_only
- wavlm_only
- ensemble

For each target, the evaluation utterance (from the manifest) is used as the
clean reference. For each selected background speaker, a random utterance
is selected from that speaker's available files.

All mixtures are generated at 0 dB SNR with specified crowd sizes.
"""
import json
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
MANIFEST_FILE = BASE_DIR / "config" / "utterance_roles.json"
SELECTIONS_DIR = BASE_DIR / "data" / "selections"
OUTPUT_DIR = BASE_DIR / "data" / "mixtures"

SR = 16000
DURATION = 12.0
TARGET_RMS = 0.1
SNR_DB = 0
CROWD_SIZES = [1, 2, 4, 6, 8]
RANDOM_SEED = 42
# ----------------------------------------------------------------

def load_audio(file_path, duration=DURATION, sr=SR):
    """
    Load audio file and trim/pad to exact duration.

    Args:
        file_path: Path to audio file
        duration: Target duration in seconds
        sr: Target sample rate

    Returns:
        np.ndarray: Audio signal at target duration and sample rate
    """
    y, fs = sf.read(file_path)
    if fs != sr:
        import librosa
        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
    target_samples = int(duration * sr)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]
    return y

def rms(x):
    """Compute root mean square of a signal."""
    return np.sqrt(np.mean(np.square(x)))

def scale_to_snr(target, bg, snr_db):
    """
    Scale background signal to achieve target SNR.

    Args:
        target: Target signal
        bg: Background signal
        snr_db: Desired SNR in decibels

    Returns:
        np.ndarray: Scaled background signal
    """
    rms_t = rms(target)
    rms_bg = rms(bg)
    if rms_bg < 1e-10:
        return np.zeros_like(bg)
    desired_rms_bg = rms_t / (10 ** (snr_db / 20.0))
    scale = desired_rms_bg / rms_bg
    return bg * scale

def normalize(y, target_rms=TARGET_RMS):
    """
    Normalise signal to target RMS level.

    Args:
        y: Input signal
        target_rms: Desired RMS level

    Returns:
        np.ndarray: Normalised signal
    """
    r = rms(y)
    if r > 0:
        return y * (target_rms / r)
    return y

def load_background_utterance(bg_spk, background_audio):
    """
    Load a random utterance from a background speaker.

    Args:
        bg_spk: Background speaker ID
        background_audio: Dictionary mapping speaker IDs to file lists

    Returns:
        np.ndarray or None: Loaded audio or None if no files available
    """
    bg_files = background_audio.get(bg_spk, [])
    if not bg_files:
        return None
    bg_path = random.choice(bg_files)
    return load_audio(bg_path)

def main():
    print("=" * 60)
    print("GENERATE VOICE CAMOUFLAGE MIXTURES")
    print("=" * 60)

    # Load the utterance role manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    targets = manifest["targets"]
    background_audio = manifest["backgrounds"]

    # Load the selection lists for each condition
    selections = {}
    for cond_name in ["random", "ecapa_only", "wavlm_only", "ensemble"]:
        selection_file = SELECTIONS_DIR / f"{cond_name}_backgrounds.json"
        if not selection_file.exists():
            print(f"⚠️ Warning: {selection_file} not found. Skipping {cond_name}.")
            continue
        with open(selection_file) as f:
            selections[cond_name] = json.load(f)

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create output directories for each condition
    dirs = {
        "random_rep1": OUTPUT_DIR / "random_rep1",
        "random_rep2": OUTPUT_DIR / "random_rep2",
        "random_rep3": OUTPUT_DIR / "random_rep3",
        "ecapa_only": OUTPUT_DIR / "ecapa_only",
        "wavlm_only": OUTPUT_DIR / "wavlm_only",
        "ensemble": OUTPUT_DIR / "ensemble",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Process each target speaker
    for spk, roles in tqdm(targets.items(), desc="Processing targets"):
        # Get the evaluation utterance for this target
        eval_path = roles["evaluation"][0]
        target_clean = load_audio(eval_path)

        # Save clean reference for each condition directory
        for d in dirs.values():
            sf.write(d / f"{spk}_clean.wav", target_clean, SR)

        # Generate mixtures for each condition
        for cond_name, cond_data in selections.items():
            if cond_name == "random":
                # Random has 3 independent repetitions
                for rep in range(1, 4):
                    rep_key = f"rep_{rep}"
                    rep_selections = cond_data.get(spk, {}).get(rep_key, {})
                    out_dir = dirs[f"random_rep{rep}"]

                    for k in CROWD_SIZES:
                        top_key = f"top_{k}"
                        bg_speakers = rep_selections.get(top_key, [])
                        if len(bg_speakers) != k:
                            continue

                        # Sum all background signals
                        bg_sum = np.zeros_like(target_clean)
                        for bg_spk in bg_speakers:
                            bg_chunk = load_background_utterance(bg_spk, background_audio)
                            if bg_chunk is None:
                                continue
                            bg_scaled = scale_to_snr(target_clean, bg_chunk, SNR_DB)
                            bg_sum += bg_scaled

                        # Combine target and background, then normalise
                        mixture = target_clean + bg_sum
                        mixture = normalize(mixture)
                        sf.write(out_dir / f"{spk}_{top_key}.wav", mixture, SR)
            else:
                # Non-random conditions (ecapa_only, wavlm_only, ensemble)
                cond_selections = cond_data.get(spk, {})
                out_dir = dirs[cond_name]

                for k in CROWD_SIZES:
                    top_key = f"top_{k}"
                    bg_speakers = cond_selections.get(top_key, [])
                    if len(bg_speakers) != k:
                        continue

                    # Sum all background signals
                    bg_sum = np.zeros_like(target_clean)
                    for bg_spk in bg_speakers:
                        bg_chunk = load_background_utterance(bg_spk, background_audio)
                        if bg_chunk is None:
                            continue
                        bg_scaled = scale_to_snr(target_clean, bg_chunk, SNR_DB)
                        bg_sum += bg_scaled

                    # Combine target and background, then normalise
                    mixture = target_clean + bg_sum
                    mixture = normalize(mixture)
                    sf.write(out_dir / f"{spk}_{top_key}.wav", mixture, SR)

    print("\n✅ All mixtures generated successfully!")
    print(f"Output directories: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()