"""
Transcribe Clean Utterances

This script transcribes clean (unmixed) evaluation utterances using Whisper
and caches the transcripts in a JSON file. The cached transcripts are used
as reference transcriptions for WER computation in subsequent steps.

This script is designed to be run locally once before uploading to Colab
for the chunked WER evaluation. It separates the computationally expensive
clean transcription from the mixture WER evaluation.
"""
import json
import whisper
import soundfile as sf
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np

# ------------------------- Configuration -------------------------
BASE_DIR = Path(__file__).parent.parent
MANIFEST_FILE = BASE_DIR / "config" / "utterance_roles.json"
OUTPUT_DIR = BASE_DIR / "results" / "wer_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "clean_transcripts.json"

SR = 16000
WHISPER_MODEL = "medium"
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
    print("=" * 60)
    print("TRANSCRIBE CLEAN UTTERANCES")
    print("=" * 60)

    # Load the utterance role manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    targets = manifest["targets"]

    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = whisper.load_model(WHISPER_MODEL, device=device)

    # Transcribe each target's clean evaluation utterance
    transcripts = {}
    for spk in tqdm(targets, desc="Transcribing clean utterances"):
        clean_path = targets[spk]["evaluation"][0]
        transcripts[spk] = transcribe_audio(clean_path, model)

    # Save transcripts to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(transcripts, f, indent=2)

    print(f"\n✅ Clean transcripts saved to {OUTPUT_FILE}")
    print(f"   Speakers: {len(transcripts)}")

if __name__ == "__main__":
    main()