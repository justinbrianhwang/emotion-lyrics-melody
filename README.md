# Emotion → Lyrics → Melody (KOR/ENG) — LSTM vs Transformer

End‑to‑end research prototype that **predicts sentiment** from user text (Korean/English), **generates emotion‑conditioned lyrics**, and **composes a monophonic melody** rendered to WAV. Each stage supports an LSTM and a Transformer variant for apples‑to‑apples comparison.

> TL;DR
> ```bash
> # 1) Prepare datasets & tokenizer
> python data_prep.py --data_dir ./data --download --spm_vocab_size 8000
>
> # 2) Train three models (choose lstm or transformer)
> python train.py train_sentiment --data_dir ./data --spm_model ./spm/bpe8k.model --out_dir ./outputs --model_type transformer
> python train.py train_lyrics    --data_dir ./data --spm_model ./spm/bpe8k.model --out_dir ./outputs --model_type transformer
> python train.py train_music     --data_dir ./data                         --out_dir ./outputs --model_type transformer
>
> # 3) End‑to‑end inference (predict → lyrics → melody → WAV)
> python train.py infer --input_text "오늘 하루 좀 힘들었어" >   --sentiment_lstm ./outputs/sentiment_lstm.pth >   --sentiment_tr   ./outputs/sentiment_transformer.pth >   --lyrics_lstm    ./outputs/lyrics_lstm.pth >   --lyrics_tr      ./outputs/lyrics_transformer.pth >   --music_lstm     ./outputs/music_lstm.pth >   --music_tr       ./outputs/music_transformer.pth >   --out_dir ./outputs
> ```

## Features
- **Bilingual tokenization** via SentencePiece BPE (default 8k) with user tokens: `<KOR>`, `<ENG>`, `<LYR>`, `<SENTI>`, `<EMO:HAPPY|SAD|ANGRY|CALM>`.
- **Reproducible data pipeline**: Kaggle download → light cleaning → JSONL splits → JSB chorales → melody sequences.
- **Switchable backbones** for each task: LSTM or Transformer.
- **Minimal synthesizer** renders melody to `song_*.wav` (no external DAW required).

## Repository Structure
```
.
├── data_prep.py              # Dataset download & preprocessing (Kaggle); trains SentencePiece
├── train.py                  # Train/evaluate/infer for sentiment, lyrics LM, melody LM
├── colab_pipeline.ipynb      # (Optional) Colab notebook for demo
├── spm/                      # SentencePiece model goes here (e.g., bpe8k.model)
├── data/                     # Prepared datasets (JSONL, JSB melody JSON)
├── outputs/                  # Checkpoints (*.pth) and generated WAV files
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Requirements
- Python **3.9+**
- See `requirements.txt` for Python packages.
- Kaggle CLI configured (`KAGGLE_USERNAME`, `KAGGLE_KEY`).

### Install
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation
The script downloads default datasets and builds train/valid/test JSONL plus a shared SentencePiece model.

**Default Kaggle slugs**
- IMDB reviews: `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`
- NSMC (Naver): `soohyun/naver-movie-review-dataset`
- Genius lyrics with language: `carlosgdcj/genius-song-lyrics-with-language-information`
- JSB chorales: `cristeapetrutheodor/jsb-chorales`

**Run**
```bash
export KAGGLE_USERNAME=...; export KAGGLE_KEY=...   # or set via kaggle.json
python data_prep.py --data_dir ./data --download --spm_vocab_size 8000 --max_lyrics_per_lang 5000
```
**Outputs (key files)**
```
data/
 ├─ sentiment/{train,valid,test}.jsonl
 ├─ lyrics/{train,valid,test}.jsonl
 └─ music/jsb_melody.json
spm/bpe8k.model
data/manifest.json
```

## Training
Each subcommand accepts shared flags like `--epochs`, `--batch_size`, `--d_model`, `--n_layers`, etc.

**Sentiment (binary: NEGATIVE/POSITIVE)**
```bash
python train.py train_sentiment --data_dir ./data --spm_model ./spm/bpe8k.model   --out_dir ./outputs --model_type transformer  # or lstm
# → outputs/sentiment_transformer.pth (or sentiment_lstm.pth)
```

**Lyrics Language Model**
```bash
python train.py train_lyrics --data_dir ./data --spm_model ./spm/bpe8k.model   --out_dir ./outputs --model_type transformer   # or lstm
# → outputs/lyrics_transformer.pth (or lyrics_lstm.pth)
```

**Melody Model (JSB chorales → monophonic note sequence)**
```bash
python train.py train_music --data_dir ./data --out_dir ./outputs --model_type transformer  # or lstm
# → outputs/music_transformer.pth (or music_lstm.pth)
```

## End‑to‑End Inference
Predicts sentiment → maps to coarse emotion → generates lyrics → generates melody → renders WAV.
```bash
python train.py infer --input_text "I feel hopeful today."   --sentiment_lstm ./outputs/sentiment_lstm.pth   --sentiment_tr   ./outputs/sentiment_transformer.pth   --lyrics_lstm    ./outputs/lyrics_lstm.pth   --lyrics_tr      ./outputs/lyrics_transformer.pth   --music_lstm     ./outputs/music_lstm.pth   --music_tr       ./outputs/music_transformer.pth   --out_dir ./outputs
# Produces: ./outputs/song_lstm.wav and ./outputs/song_tr.wav
```

### Notes
- Language/conditioning tokens are inserted automatically (`<KOR>/<ENG>`, `<SENTI>`, `<LYR>`, and `<EMO:*>`).
- Checkpoints embed compact **training subsets** for reproducibility and quick demos.
- Melody rendering uses a simple oscillator; BPM is adjusted by emotion (e.g., HAPPY ≈ 100 BPM).

## Tips for Reproducibility
- Seeds are fixed to 42 across NumPy/PyTorch.
- Use the same `spm/bpe8k.model` across tasks to ensure consistent token IDs.
- GPU optional; code auto‑selects CUDA if available.

## Acknowledgements
This repository prepares data from publicly available Kaggle datasets (see slugs above). Please check each dataset’s original license.

## License
MIT — see [LICENSE](LICENSE).
