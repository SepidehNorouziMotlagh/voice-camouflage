# Voice Camouflage: Model-Dependent Speaker Selection

## Overview

This project investigates whether selecting acoustically dissimilar interfering speakers can increase speaker-verification difficulty when a target utterance is mixed with multiple background voices.

It compares three targeted speaker-selection strategies against random selection:

- **ECAPA-only:** Selects background speakers with the lowest ECAPA-TDNN cosine similarity to the target.
- **WavLM-only:** Selects background speakers with the lowest WavLM-XVector cosine similarity.
- **Ensemble:** Combines normalized similarity ranks from both models.
- **Random:** Selects background speakers randomly, with three repetitions.

The evaluation examines speaker-verification EER using two pretrained speaker-recognition systems, alongside transcription-based and signal-level measures of speech utility.

This is an exploratory study of model-dependent speaker selection under synthetic multi-speaker interference. Its results should not be interpreted as a general guarantee of speaker anonymity.

## Dataset

The experiments use the **LibriSpeech train-clean-100** subset.

The speaker partition used in the recorded experiment contains:

- 100 target speakers
- 151 background speakers

For each target speaker, the available utterances are divided into:

| Role | Utterances per target | Purpose |
|---|---:|---|
| Selection | 2 | Compute the target speaker representation |
| Enrollment | 1 | Enroll the target in the speaker-verification evaluation |
| Evaluation | 1 | Generate clean references and mixtures |

The target speaker representation is calculated by averaging embeddings from the two selection utterances. Background speaker representations are calculated from the first available utterance for each background speaker.

The original audio files are not included in this repository. Download LibriSpeech separately from [OpenSLR](https://www.openslr.org/12).

## Method

### 1. Speaker representation and similarity scoring

Two pretrained speaker-recognition models are used:

- **ECAPA-TDNN:** `speechbrain/spkrec-ecapa-voxceleb`
- **WavLM-XVector:** `microsoft/wavlm-base-plus-sv`

Cosine similarity is calculated between each target representation and each background speaker representation.

Lower similarity scores are treated as greater speaker dissimilarity.

### 2. Background speaker selection

Four selection conditions are evaluated:

| Strategy | Selection procedure |
|---|---|
| Random | Random background speaker selection, three repetitions |
| ECAPA-only | Selects speakers with the lowest ECAPA similarity |
| WavLM-only | Selects speakers with the lowest WavLM similarity |
| Ensemble | Selects speakers using the average of normalized ECAPA and WavLM ranks |

The ensemble strategy uses rank fusion rather than directly averaging similarity scores from two different models.

Five crowd sizes are considered: **1, 2, 4, 6, and 8 interfering speakers**.

### 3. Mixture generation

For each target utterance, the selected background speakers contribute one randomly chosen utterance each.

The mixture-generation configuration uses:

- Sample rate: 16 kHz
- Mixture duration: 12 seconds
- Target RMS normalization: 0.1
- Individual interferer scaling: 0 dB relative to the target

Each background utterance is independently scaled to the target's RMS before the background signals are summed with the target.

**Important:** The 0 dB setting applies to each individual interferer, not to the combined background crowd. The overall target-to-interference ratio therefore changes with the number of interfering speakers.

The resulting mixture is normalized to the configured target RMS before being saved.

The pipeline generates up to 3,000 mixture files across six conditions and five crowd sizes, using the 100 target speakers.

## Evaluation

### Speaker-verification privacy proxy: EER

Equal Error Rate (EER) is calculated separately using ECAPA-TDNN and WavLM-XVector.

For each target mixture, the evaluation constructs:

- A genuine trial using the target's enrollment utterance and the mixed utterance.
- An impostor trial using an enrollment utterance from one randomly selected different target speaker and the same mixture.

**Aggregation detail:** The current EER scripts pool trials from all five crowd sizes before calculating one EER per condition. Consequently, the reported EER values do not represent the four-interferer condition alone.

Higher EER can indicate greater difficulty for the verifier under its specified score direction. However, EER values above 0.5 require additional score-direction and ROC analysis before being interpreted as stronger privacy. EER is a model-specific evaluation measure, not proof of anonymization.

### Transcription-based utility: WER

Whisper Medium is used to transcribe clean target utterances and their corresponding mixtures.

The transcript generated from the clean utterance is used as the reference when computing Word Error Rate (WER) for the mixture transcript.

Therefore, these values measure transcription differences relative to Whisper's clean-audio output. They are not WER values against LibriSpeech's original human reference transcripts.

### Signal-level utility: STOI

Short-Time Objective Intelligibility (STOI) is calculated between the clean target utterance and its corresponding mixture.

STOI provides a signal-level intelligibility measure. It does not measure speaker identity protection.

### Results aggregation

The final summary table combines:

- EER values pooled across all five crowd sizes.
- WER values at four interfering speakers.
- STOI values at four interfering speakers.

The columns therefore represent different aggregation scopes and should not be interpreted as measurements from one identical crowd-size condition.

## Results

### Speaker-verification evaluation

<small>One EER value per condition, pooled across crowd sizes. The Random row is the arithmetic mean of the three random repetitions.</small>

| Selection strategy | ECAPA EER | WavLM EER |
|---|---:|---:|
| Random (mean of 3 repetitions) | 0.320 | 0.405 |
| ECAPA-only | 0.450 | 0.414 |
| WavLM-only | 0.494 | 0.652 |
| Ensemble | **0.564** | 0.642 |

The ensemble strategy produces the highest reported ECAPA EER. WavLM-only selection produces a slightly higher WavLM EER than the ensemble strategy.

These results illustrate that selection effectiveness depends on the speaker-verification backend.

### Utility evaluation at four interfering speakers

<small>Random values are averaged across the three random repetitions. WER uses Whisper's clean-audio transcript as its reference.</small>

| Selection strategy | Whisper-reference WER | STOI |
|---|---:|---:|
| Random (mean of 3 repetitions) | 1.190 | 0.524 |
| ECAPA-only | 1.212 | 0.517 |
| WavLM-only | **1.059** | 0.528 |
| Ensemble | 1.095 | 0.524 |

The utility measurements show differences between selection strategies, although the STOI values are relatively close. The WER values should be interpreted as transcription differences relative to the clean Whisper output, rather than conventional reference-based ASR performance.

## Repository Structure

```text
voice-camouflage/
├── config/
│   └── speaker_partition.json
├── scripts/
│   ├── 00_build_manifest.py
│   ├── 01_collect_paths.py
│   ├── 02a_compute_ecapa_scores.py
│   ├── 02b_compute_wavlm_scores.py
│   ├── 02c_merge_scores.py
│   ├── 03_generate_selections.py
│   ├── 04_generate_mixtures.py
│   ├── 05a_evaluate_eer_ecapa.py
│   ├── 05b_evaluate_eer_wavlm.py
│   ├── 05c_merge_eer.py
│   ├── 06a_transcribe_clean.py
│   ├── 06b_wer_condition_chunked.py
│   ├── 06c_merge_condition_chunks.py
│   ├── 06d_merge_all_conditions.py
│   ├── 07_evaluate_stoi.py
│   ├── 08_generate_summary_table.py
│   └── 09_generate_plots.py
├── results/
│   ├── embeddings/
│   ├── eer_results/
│   ├── wer_results/
│   ├── stoi_results/
│   └── final_results_table.csv
├── plots/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The audio dataset, generated path indexes, utterance manifest, selection lists, and mixtures are local generated files and are not included in the repository.

## Reproducing the Pipeline

### 1. Clone the repository

```bash
git clone https://github.com/SepidehNorouziMotlagh/voice-camouflage.git
cd voice-camouflage
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download LibriSpeech

Download `train-clean-100.tar.gz` from [OpenSLR](https://www.openslr.org/12).

Extract it to:

```text
data/LibriSpeech/train-clean-100/
```

### 4. Generate audio path indexes

The committed `config/speaker_partition.json` preserves the target/background speaker assignment from the experiment.

Run:

```bash
python scripts/01_collect_paths.py
```

This generates:

```text
data/target_audio_paths.json
data/background_audio_paths.json
```

### 5. Build the utterance manifest

```bash
python scripts/00_build_manifest.py
```

This creates the local manifest:

```text
config/utterance_roles.json
```

### 6. Compute speaker similarities

```bash
python scripts/02a_compute_ecapa_scores.py
python scripts/02b_compute_wavlm_scores.py
python scripts/02c_merge_scores.py
```

### 7. Generate speaker selections

```bash
python scripts/03_generate_selections.py
```

### 8. Generate mixtures

```bash
python scripts/04_generate_mixtures.py
```

### 9. Evaluate speaker-verification EER

```bash
python scripts/05a_evaluate_eer_ecapa.py
python scripts/05b_evaluate_eer_wavlm.py
python scripts/05c_merge_eer.py
```

### 10. Evaluate transcription utility

First, generate clean Whisper transcripts:

```bash
python scripts/06a_transcribe_clean.py
```

The chunked WER scripts are configured for Google Colab. Update `BASE_DIR` in scripts `06b`, `06c`, and `06d` to point to the project directory in Google Drive.

For each condition, run the ten speaker chunks:

```bash
python scripts/06b_wer_condition_chunked.py random_rep1 0
python scripts/06b_wer_condition_chunked.py random_rep1 1
```

Continue through chunk 9, then merge:

```bash
python scripts/06c_merge_condition_chunks.py random_rep1
```

Repeat for the other five conditions and run:

```bash
python scripts/06d_merge_all_conditions.py
```

### 11. Evaluate STOI and generate summaries

```bash
python scripts/07_evaluate_stoi.py
python scripts/08_generate_summary_table.py
python scripts/09_generate_plots.py
```

## Limitations and Reproducibility Notes

- The evaluation uses two pretrained speaker-verification backends; results may not generalize to other systems.
- EER is pooled across the five crowd sizes in the current implementation.
- Each impostor trial uses one randomly selected different target speaker rather than an exhaustive set of impostor comparisons.
- EER values above 0.5 require score-direction analysis before being interpreted as privacy improvements.
- Each interferer is independently scaled to 0 dB relative to the target. The combined crowd does not have a fixed overall SNR.
- The current random-selection implementation resets its random seed inside the target-speaker loop. This causes the same background-speaker selections to be reused across targets within a given repetition.
- WER is calculated against Whisper transcripts of clean utterances, not human reference transcripts.
- STOI measures signal-level intelligibility and is not a privacy metric.
- The current results do not establish resistance to adaptive attackers, speaker re-identification using other models, or a general privacy guarantee.

Future work should examine fixed overall mixture SNR, independent random-selection trials, per-crowd-size EER, more extensive impostor trials, and score-direction-aware verification analysis.

## Dataset Citation

Please cite the original LibriSpeech publication when using the dataset:

Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). *LibriSpeech: An ASR Corpus Based on Public Domain Audio Books*. 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5206–5210.

https://doi.org/10.1109/ICASSP.2015.7178964

## License

MIT License. See `LICENSE`.

## Author

**Sepideh Norouzi Motlagh**

[GitHub](https://github.com/SepidehNorouziMotlagh)
