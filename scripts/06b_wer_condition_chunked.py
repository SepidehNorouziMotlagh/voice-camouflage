"""
Compute WER for a Single Condition (Chunked)

This script computes Word Error Rate (WER) for a specific condition and speaker
chunk. It processes a subset of speakers (chunk) to enable parallel execution
or resumable processing on cloud GPUs.

Usage:
    python 06b_wer_condition_chunked.py <condition> <chunk_id>

Example:
    python 06b_wer_condition_chunked.py random_rep1 0

Available conditions:
    random_rep1, random_rep2, random_rep3, ecapa_only, wavlm_only, ensemble

Chunk IDs: 0..9 (for 10 speakers per chunk, total 100 speakers)
"""
import json
import whisper
import soundfile as sf
from pathlib import Path
import torch
import jiwer
from tqdm import tqdm
import sys
import pandas as pd

# ------------------------- Configuration -------------------------
# IMPORTANT: This script is designed to run on Google Colab.
# Update BASE_DIR to match your Google Drive path.
BASE_DIR = Path("/content/drive/MyDrive/voice_camouflage")
MANIFEST_FILE = BASE_DIR / "config" / "utterance_roles.json"
MIXTURES_DIR = BASE_DIR / "data" / "mixtures"
OUTPUT_DIR = BASE_DIR / "results" / "wer_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_TRANSCRIPTS = OUTPUT_DIR / "clean_transcripts.json"

SR = 16000
WHISPER_MODEL = "medium"
CROWD_SIZES = [1, 2, 4, 6, 8]
CHUNK_SIZE = 10
# ----------------------------------------------------------------

def transcribe_audio(audio_path, model):
    """
    Transcribe audio file using Whisper.

    Args:
        audio_path: Path to audio file
        model: Loaded Whisper model

    Returns:
        str: Transcript text
    """
    try:
        y, fs = sf.read(audio_path)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if fs != SR:
            import librosa
            y = librosa.resample(y.astype(np.float32), orig_sr=fs, target_sr=SR)
        y = y.astype(np.float32)
        result = model.transcribe(y, fp16=False)
        return result["text"].strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def main():
    # Check command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python 06b_wer_condition_chunked.py <condition> <chunk_id>")
        print("Example: python 06b_wer_condition_chunked.py random_rep1 0")
        print("Available conditions: random_rep1, random_rep2, random_rep3, ecapa_only, wavlm_only, ensemble")
        print("Chunk IDs: 0..9 (for 10 speakers per chunk)")
        sys.exit(1)

    condition = sys.argv[1]
    chunk_id = int(sys.argv[2])

    print("=" * 60)
    print(f"WER EVALUATION - {condition} (chunk {chunk_id})")
    print("=" * 60)

    # Check that clean transcripts exist
    if not CLEAN_TRANSCRIPTS.exists():
        print(f"❌ Clean transcripts not found: {CLEAN_TRANSCRIPTS}")
        print("   Please run 06a_transcribe_clean.py first.")
        sys.exit(1)

    # Load manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    targets = manifest["targets"]
    speaker_list = list(targets.keys())

    # Split speakers into chunks
    chunks = [speaker_list[i:i+CHUNK_SIZE] for i in range(0, len(speaker_list), CHUNK_SIZE)]
    if chunk_id >= len(chunks):
        print(f"Chunk {chunk_id} not available. Max chunk: {len(chunks)-1}")
        sys.exit(1)

    chunk_speakers = chunks[chunk_id]
    print(f"Processing {len(chunk_speakers)} speakers (chunk {chunk_id}/{len(chunks)-1})")

    # Load cached clean transcripts
    with open(CLEAN_TRANSCRIPTS) as f:
        clean_transcripts = json.load(f)

    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = whisper.load_model(WHISPER_MODEL, device=device)

    results = []
    cond_dir = MIXTURES_DIR / condition

    for spk in tqdm(chunk_speakers, desc=f"Chunk {chunk_id}"):
        clean_text = clean_transcripts.get(spk, "")
        if not clean_text:
            continue

        for k in CROWD_SIZES:
            mix_path = cond_dir / f"{spk}_top_{k}.wav"
            if not mix_path.exists():
                continue

            mix_text = transcribe_audio(mix_path, model)
            wer = jiwer.wer(clean_text, mix_text)

            results.append({
                "speaker": spk,
                "condition": condition,
                "crowd_size": k,
                "WER": wer
            })

    # Save chunk results
    df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / f"wer_{condition}_chunk{chunk_id}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Chunk {chunk_id} for {condition} saved to {output_file}")
    print(f"   Speakers: {len(chunk_speakers)}, Samples: {len(results)}")

    # Print summary for this chunk
    if len(results) > 0:
        agg = df.groupby(["crowd_size"])["WER"].agg(["mean", "std", "count"]).reset_index()
        print("\nSummary (chunk):")
        print(agg.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    main()