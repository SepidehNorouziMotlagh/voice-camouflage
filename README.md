# Voice Camouflage: Model-Dependent Speaker Selection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📋 Overview

This repository contains the complete code and experimental pipeline for the paper:

> **"Model-Dependent Speaker Selection for Voice Camouflage: A Study of Privacy-Utility Trade-offs and Cross-Model Robustness"**

Voice camouflage protects speaker privacy by mixing a target utterance with interfering speech. This project investigates whether **targeted selection of interfering speakers** improves privacy and whether the effect transfers across speaker verification models.

### Key Findings

- **Ensemble selection** (ECAPA + WavLM rank fusion) achieves the best cross-model privacy:
  - ECAPA EER: **0.564**
  - WavLM EER: **0.642**
- **Selection effectiveness is model-dependent**: ECAPA-only selection overfits to ECAPA and shows limited transfer to WavLM
- **Strongest effect at 4 interfering speakers** with diminishing returns at larger crowd sizes
- **Utility preserved**: WER and STOI remain competitive with random baselines

## 📁 Repository Structure

```
voice-camouflage/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── config/
│   └── utterance_roles.json          # Utterance splits for each speaker
│
├── scripts/
│   ├── 00_build_manifest.py          # Build utterance role manifest
│   ├── 02a_compute_ecapa_scores.py   # ECAPA similarity computation
│   ├── 02b_compute_wavlm_scores.py   # WavLM similarity computation
│   ├── 02c_merge_scores.py           # Merge ECAPA and WavLM scores
│   ├── 03_generate_selections.py     # Generate selection lists
│   ├── 04_generate_mixtures.py       # Generate voice camouflage mixtures
│   ├── 05a_evaluate_eer_ecapa.py     # EER evaluation (ECAPA-TDNN)
│   ├── 05b_evaluate_eer_wavlm.py     # EER evaluation (WavLM-XVector)
│   ├── 05c_merge_eer.py              # Merge EER results
│   ├── 06a_transcribe_clean.py       # Transcribe clean utterances
│   ├── 06b_wer_condition_chunked.py  # WER for a condition (chunked)
│   ├── 06c_merge_condition_chunks.py # Merge WER chunks for a condition
│   ├── 06d_merge_all_conditions.py   # Merge all WER conditions
│   ├── 07_evaluate_stoi.py           # STOI evaluation
│   ├── 08_generate_summary_table.py  # Generate final results table
│   └── 09_generate_plots.py          # Generate publication plots
│
├── results/
│   ├── embeddings/                   # ECAPA and WavLM similarity scores
│   ├── eer_results/                  # EER results under both models
│   ├── wer_results/                  # WER results and transcripts
│   ├── stoi_results/                 # STOI results
│   └── final_results_table.csv       # Combined results table
│
├── plots/
│   ├── wer_vs_crowd.png
│   ├── stoi_vs_crowd.png
│   ├── tradeoff_wavlm_vs_wer.png
│   └── bar_chart_4talkers.png)
```

## 🧪 Experimental Pipeline

### 1. Data Preparation

```
00_build_manifest.py
```
- Splits utterances into selection, enrollment, and evaluation sets
- Creates `config/utterance_roles.json`

### 2. Embedding Computation

```
02a_compute_ecapa_scores.py  → ECAPA similarity matrix
02b_compute_wavlm_scores.py  → WavLM similarity matrix
02c_merge_scores.py          → Combined similarity matrix
```

### 3. Selection List Generation

```
03_generate_selections.py
```
- Creates selection lists for four strategies:
  - Random (3 repetitions)
  - ECAPA-only
  - WavLM-only
  - Ensemble

### 4. Mixture Generation

```
04_generate_mixtures.py
```
- Generates 3,000 mixture files across:
  - 6 conditions (random_rep1–3, ecapa_only, wavlm_only, ensemble)
  - 5 crowd sizes (1, 2, 4, 6, 8)

### 5. Privacy Evaluation (EER)

```
05a_evaluate_eer_ecapa.py  → ECAPA-TDNN EER
05b_evaluate_eer_wavlm.py  → WavLM-XVector EER
05c_merge_eer.py           → Combined EER results
```

### 6. Utility Evaluation

```
06a_transcribe_clean.py       → Cache clean transcripts
06b_wer_condition_chunked.py  → WER (chunked for Colab)
06c_merge_condition_chunks.py → Merge WER chunks
06d_merge_all_conditions.py   → Merge all conditions
07_evaluate_stoi.py           → STOI evaluation
```

### 7. Results and Visualisation

```
08_generate_summary_table.py  → Final results table
09_generate_plots.py          → Publication plots
```

## 🚀 Requirements

### Installation

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
speechbrain>=1.0.0
transformers>=4.36.0
torch>=2.0.0
torchaudio>=2.0.0
soundfile>=0.12.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
pystoi>=0.3.0
openai-whisper>=20231117
jiwer>=3.0.0
tqdm>=4.65.0
```

## 🛠️ How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/voice-camouflage.git
cd voice-camouflage
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download LibriSpeech

Download `train-clean-100.tar.gz` from [OpenSLR](https://www.openslr.org/12/).

### 4. Update Paths

In the scripts, update the `LIBRI_PATH` variable to point to your LibriSpeech directory.

### 5. Run the Pipeline

Run the scripts in numerical order:

```bash
python scripts/00_build_manifest.py
python scripts/02a_compute_ecapa_scores.py
python scripts/02b_compute_wavlm_scores.py
# ... and so on
```

### 6. Running WER on Google Colab

For WER evaluation, the chunked scripts are designed for Google Colab:

1. Upload the project folder to Google Drive
2. Update `BASE_DIR` in the scripts to match your Drive path
3. Run chunks sequentially:

```bash
python scripts/06a_transcribe_clean.py
python scripts/06b_wer_condition_chunked.py random_rep1 0
python scripts/06b_wer_condition_chunked.py random_rep1 1
# ... up to 9
python scripts/06c_merge_condition_chunks.py random_rep1
# Repeat for other conditions
python scripts/06d_merge_all_conditions.py
```

## 📊 Results

### Privacy (EER)

| Selection Strategy | ECAPA EER | WavLM EER |
| ------------------ | --------: | --------: |
| Random             |     0.320 |     0.405 |
| ECAPA-only         |     0.450 |     0.414 |
| WavLM-only         |     0.494 |     0.652 |
| **Ensemble**       | **0.564** | **0.642** |

### Utility (4 Interfering Speakers)

| Selection Strategy |   WER |  STOI |
| ------------------ | ----: | ----: |
| Random             | 1.619 | 0.524 |
| ECAPA-only         | 1.212 | 0.517 |
| WavLM-only         | 1.059 | 0.528 |
| **Ensemble**       | 1.095 | 0.524 |

### Key Takeaways

1. **Ensemble selection** provides the most consistent cross-model privacy
2. **Selection effectiveness is model-dependent** (ECAPA-only overfits)
3. **Strongest effect at 4 interfering speakers**
4. **Utility preserved** (WER and STOI remain competitive)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{motlagh2025voicecamouflage,
  title={Model-Dependent Speaker Selection for Voice Camouflage: A Study of Privacy-Utility Trade-offs and Cross-Model Robustness},
  author={Motlagh, Sepideh Norouzi},
  year={2025}
}
```

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [LibriSpeech](https://www.openslr.org/12/) for the read speech corpus
- [SpeechBrain](https://speechbrain.github.io/) for the ECAPA-TDNN model
- [Hugging Face](https://huggingface.co/) for the WavLM model and transformers
- [Whisper](https://github.com/openai/whisper) for ASR transcription
- [PySTOI](https://github.com/mpariente/pystoi) for intelligibility evaluation
- [JiWER](https://github.com/jitsi/jiwer) for WER computation
```