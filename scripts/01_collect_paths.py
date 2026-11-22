
import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PARTITION_FILE = BASE_DIR / "config" / "speaker_partition.json"
OUTPUT_DIR = BASE_DIR / "data"


def collect_speaker_files(dataset_root: Path, speaker_ids: list[str]) -> dict:
    collected = {}
    missing = []

    for speaker_id in speaker_ids:
        speaker_dir = dataset_root / str(speaker_id)

        files = sorted(speaker_dir.rglob("*.flac"))
        if not files:
            missing.append(str(speaker_id))
            continue

        collected[str(speaker_id)] = [
            str(path.resolve()) for path in files
        ]

    if missing:
        raise FileNotFoundError(
            "No LibriSpeech FLAC files found for speaker IDs: "
            + ", ".join(missing)
        )

    return collected


def main():
    parser = argparse.ArgumentParser(
        description="Build LibriSpeech target/background audio path indexes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=BASE_DIR / "data" / "LibriSpeech" / "train-clean-100",
        help="Path to the extracted LibriSpeech train-clean-100 directory.",
    )
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {args.dataset_root}"
        )

    partition = json.loads(
        PARTITION_FILE.read_text(encoding="utf-8")
    )

    targets = [str(x) for x in partition["target_speakers"]]
    backgrounds = [str(x) for x in partition["background_speakers"]]

    overlap = set(targets) & set(backgrounds)
    if overlap:
        raise ValueError(
            f"Target/background speaker overlap detected: {sorted(overlap)}"
        )

    target_paths = collect_speaker_files(args.dataset_root, targets)
    background_paths = collect_speaker_files(args.dataset_root, backgrounds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "target_audio_paths.json").write_text(
        json.dumps(target_paths, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "background_audio_paths.json").write_text(
        json.dumps(background_paths, indent=2),
        encoding="utf-8",
    )

    print(f"Target speakers: {len(target_paths)}")
    print(f"Background speakers: {len(background_paths)}")
    print(f"Path indexes saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
