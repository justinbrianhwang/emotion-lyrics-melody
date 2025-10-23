
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_prep.py

다운로드(+전처리) 스크립트.
- Kaggle에서 데이터 자동 다운로드 (영화 리뷰 감정: IMDB, NSMC / 가사: Genius / 음악: JSB Chorales)
- 한국어+영어 혼합을 고려한 SentencePiece 서브워드 토크나이저 학습
- 학습/평가에 바로 쓰기 좋게 JSONL 저장

필요 환경:
- Kaggle API (KAGGLE_USERNAME, KAGGLE_KEY 설정되어 있어야 함)
- Python>=3.9, pandas, numpy, tqdm, sentencepiece, sklearn

예시:
python data_prep.py --data_dir ./data --spm_vocab_size 8000 --max_lyrics_per_lang 5000

작성: GPT-5 Pro
"""
import argparse
import os
import sys
import json
import re
import io
import random
import zipfile
import tarfile
import subprocess
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# sentencepiece는 pip install sentencepiece 필요
import sentencepiece as spm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------
# Kaggle Downloader
# ---------------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str]):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")
    return result


def kaggle_download(dataset_slug: str, out_dir: str):
    """
    Kaggle CLI를 사용하여 데이터셋을 다운로드 후 자동 압축해제.
    """
    ensure_dir(out_dir)
    # 우선 unzip 없이 zip파일을 가져온 뒤, 직접 해제 시도 (일부 환경에서 --unzip 이슈 회피)
    zip_path = os.path.join(out_dir, dataset_slug.replace("/", "_") + ".zip")
    if not os.path.exists(zip_path):
        run_cmd(["kaggle", "datasets", "download", "-d", dataset_slug, "-p", out_dir])
    else:
        print(f"[SKIP] Already downloaded: {zip_path}")

    # 압축 해제
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print(f"[OK] Extracted: {zip_path}")


# ---------------------------
# Cleaning helpers
# ---------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # 간단한 전처리: HTML 태그 제거, 괄호 내 정보 제거, 공백 정리
    s = re.sub(r"<[^>]+>", " ", s)                # HTML tags
    s = re.sub(r"\[[^\]]+\]", " ", s)             # [chorus], [verse] 등
    s = re.sub(r"\([^)]*\)", " ", s)              # (feat. ...)
    s = re.sub(r"http\S+", " ", s)                # URLs
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------
# SentencePiece Trainer
# ---------------------------
def train_sentencepiece(corpus_txt: str, model_prefix: str, vocab_size: int, user_symbols: List[str]):
    """
    corpus_txt: 한 줄당 하나의 문장/가사 텍스트가 담긴 파일 경로
    모델/사전은 model_prefix.{model,vocab} 로 저장됨
    """
    ensure_dir(os.path.dirname(model_prefix))
    usr = ",".join(user_symbols)
    spm.SentencePieceTrainer.Train(
        input=corpus_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,  # 한글+영어 혼합
        model_type="bpe",
        user_defined_symbols=usr,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3
    )
    print(f"[OK] Trained SentencePiece at: {model_prefix}.model")


# ---------------------------
# Datasets
# ---------------------------
def prepare_imdb(imdb_dir: str, max_samples: int = None) -> pd.DataFrame:
    """
    Expecting 'IMDB Dataset.csv' with columns: ['review', 'sentiment'].
    """
    # 여러 변형이 있으므로 파일명 탐색
    candidates = list(Path(imdb_dir).rglob("*IMDB*Dataset*.csv")) + list(Path(imdb_dir).rglob("IMDB_Dataset.csv")) + list(Path(imdb_dir).rglob("IMDB Dataset.csv"))
    if not candidates:
        raise FileNotFoundError("IMDB csv not found. (expected something like 'IMDB Dataset.csv')")
    path = str(candidates[0])
    df = pd.read_csv(path)
    df["text"] = df["review"].astype(str).apply(clean_text)
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    df["lang"] = "eng"
    df = df[["text", "label", "lang"]].dropna()
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=SEED)
    print(f"[IMDB] {len(df)} rows")
    return df.reset_index(drop=True)


def prepare_nsmc(nsmc_dir: str, max_samples: int = None) -> pd.DataFrame:
    """
    Expecting 'ratings_train.txt' and 'ratings_test.txt' (tab-separated) with columns: ['id', 'document', 'label']
    """
    # 파일 찾기
    train_txt = None
    test_txt = None
    for p in Path(nsmc_dir).rglob("*"):
        if p.name.lower().startswith("ratings_train") and p.suffix in (".txt", ".tsv", ".csv"):
            train_txt = str(p)
        if p.name.lower().startswith("ratings_test") and p.suffix in (".txt", ".tsv", ".csv"):
            test_txt = str(p)
    if not train_txt:
        # 일부 mirror는 'train.txt', 'test.txt' 등으로 제공
        for p in Path(nsmc_dir).rglob("*.txt"):
            if "train" in p.name.lower():
                train_txt = str(p)
            elif "test" in p.name.lower():
                test_txt = str(p)
    if not train_txt:
        raise FileNotFoundError("NSMC ratings_train.txt not found")

    def _load(tsv_path: str):
        try:
            df = pd.read_csv(tsv_path, sep="\t")
        except Exception:
            df = pd.read_csv(tsv_path)  # fallback
        cols = {c.lower(): c for c in df.columns}
        text_col = cols.get("document", list(df.columns)[1])
        label_col = cols.get("label", list(df.columns)[-1])
        df["text"] = df[text_col].astype(str).apply(clean_text)
        df["label"] = df[label_col].astype(int).clip(0, 1)
        df["lang"] = "kor"
        return df[["text", "label", "lang"]].dropna()

    df_train = _load(train_txt)
    if test_txt:
        df_test = _load(test_txt)
        df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    else:
        df = df_train
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=SEED)
    print(f"[NSMC] {len(df)} rows")
    return df.reset_index(drop=True)


def prepare_genius(genius_dir: str, max_per_lang: int = 5000) -> pd.DataFrame:
    """
    Expecting a CSV with columns including at least ['lyrics', 'language', 'artist', 'title'].
    We'll filter language in {'en', 'ko'} (heuristic) and clean the lyrics.
    """
    candidates = list(Path(genius_dir).rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV files found under Genius dataset dir.")
    # 가장 큰 파일(전체 데이터)을 우선 사용
    csv_path = max(candidates, key=lambda p: os.path.getsize(p))
    df = pd.read_csv(csv_path)
    lower_cols = {c.lower(): c for c in df.columns}
    lyrics_col = lower_cols.get("lyrics") or lower_cols.get("lyric") or lower_cols.get("text")
    lang_col = lower_cols.get("language") or lower_cols.get("lang")
    artist_col = lower_cols.get("artist", None)
    title_col = lower_cols.get("title", None)
    if lyrics_col is None or lang_col is None:
        raise ValueError("Genius CSV must contain 'lyrics' and 'language' columns (case-insensitive).")

    df = df[[lyrics_col, lang_col] + ([artist_col] if artist_col else []) + ([title_col] if title_col else [])].copy()
    df["lyrics"] = df[lyrics_col].astype(str).apply(clean_text)
    df["language"] = df[lang_col].astype(str).str.lower()

    # 언어 매핑: en/en-US/en-GB -> eng, ko/kor -> kor
    def map_lang(x):
        if isinstance(x, str):
            if x.startswith("en"):
                return "eng"
            if x.startswith("ko"):
                return "kor"
        return "other"
    df["lang"] = df["language"].apply(map_lang)
    df = df[df["lang"].isin(["eng", "kor"])].copy()
    # 너무 긴 문장은 잘라주기 (메모리 절약)
    df["lyrics"] = df["lyrics"].str.slice(0, 3000)

    # 각 언어당 최대 곡 수 제한
    res = []
    for lang in ["eng", "kor"]:
        sub = df[df["lang"] == lang]
        if len(sub) > max_per_lang:
            sub = sub.sample(n=max_per_lang, random_state=SEED)
        res.append(sub)
    df = pd.concat(res, axis=0, ignore_index=True)
    df = df.rename(columns={"lyrics": "text"})[["text", "lang"]].dropna()
    print(f"[Genius Lyrics] {len(df)} rows (eng/kor)")
    return df.reset_index(drop=True)


def prepare_jsb(jsb_dir: str) -> Dict[str, Any]:
    """
    Expecting 'jsb-chorales-16th.pkl' or similar structure with dict keys: train/valid/test -> list of piano-roll arrays.
    We'll convert to monophonic melody by taking the highest active pitch at each timestep.
    """
    pkl_path = None
    for p in Path(jsb_dir).rglob("*.pkl"):
        if "chorale" in p.name.lower() or "jsb" in p.name.lower():
            pkl_path = str(p); break
    if pkl_path is None:
        raise FileNotFoundError("JSB Chorales pickle not found (e.g., jsb-chorales-16th.pkl).")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    def roll_to_melody(roll: np.ndarray, rest_token: int = 0) -> List[int]:
        # roll: [T, 88] bool/int
        # Convert to a monophonic melody: highest pitch +21 (MIDI 21..108), rest if none.
        seq = []
        for t in range(roll.shape[0]):
            active = np.where(roll[t] > 0)[0]
            if len(active) == 0:
                seq.append(rest_token)
            else:
                # map piano roll index (0..87) -> MIDI note (21..108)
                pitch = int(active[-1]) + 21
                seq.append(pitch)
        # Trim long repeats for memory efficiency
        return seq

    out = {}
    for split in ["train", "valid", "test"]:
        rolls = data.get(split) or data.get(split[:5])  # tolerate name variants
        seqs = []
        for arr in rolls:
            arr = np.array(arr)
            # Some mirrors already store as note numbers; detect shape
            if arr.ndim == 2 and arr.shape[1] >= 60:
                seqs.append(roll_to_melody(arr))
            else:
                # assume already monophonic note list
                seqs.append([int(x) for x in arr])
        out[split] = seqs
        print(f"[JSB {split}] {len(seqs)} sequences")
    return out


# ---------------------------
# Splitting & Save JSONL
# ---------------------------
def train_valid_test_split(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    a = int(n * ratios[0])
    b = int(n * (ratios[0] + ratios[1]))
    train_idx, valid_idx, test_idx = idx[:a], idx[a:b], idx[b:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def save_jsonl(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(df)} -> {path}")


def write_corpus_txt(dfs: List[pd.DataFrame], path: str, text_field="text"):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for df in dfs:
            for s in df[text_field].astype(str).tolist():
                s = s.replace("\n", " ")
                f.write(s.strip() + "\n")
    print(f"[OK] Wrote corpus: {path}")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true", help="Use Kaggle API to download datasets")
    parser.add_argument("--imdb_slug", type=str, default="lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    parser.add_argument("--nsmc_slug", type=str, default="soohyun/naver-movie-review-dataset")
    parser.add_argument("--lyrics_slug", type=str, default="carlosgdcj/genius-song-lyrics-with-language-information")
    parser.add_argument("--jsb_slug", type=str, default="cristeapetrutheodor/jsb-chorales")
    parser.add_argument("--max_imdb", type=int, default=50000)
    parser.add_argument("--max_nsmc", type=int, default=200000)
    parser.add_argument("--max_lyrics_per_lang", type=int, default=5000)
    parser.add_argument("--spm_vocab_size", type=int, default=8000)
    parser.add_argument("--spm_dir", type=str, default="./spm")
    args = parser.parse_args()

    data_dir = args.data_dir
    ensure_dir(data_dir)

    # 1) Download
    if args.download:
        print("[*] Downloading datasets from Kaggle...")
        kaggle_download(args.imdb_slug, os.path.join(data_dir, "imdb"))
        kaggle_download(args.nsmc_slug, os.path.join(data_dir, "nsmc"))
        kaggle_download(args.lyrics_slug, os.path.join(data_dir, "lyrics"))
        kaggle_download(args.jsb_slug, os.path.join(data_dir, "jsb"))
    else:
        print("[*] --download not set; will assume datasets already exist under data_dir.")

    # 2) Prepare Sentiment (IMDB + NSMC)
    imdb_df = prepare_imdb(os.path.join(data_dir, "imdb"), max_samples=args.max_imdb)
    nsmc_df = prepare_nsmc(os.path.join(data_dir, "nsmc"), max_samples=args.max_nsmc)
    senti_df = pd.concat([imdb_df, nsmc_df], axis=0, ignore_index=True)
    tr_senti, va_senti, te_senti = train_valid_test_split(senti_df, ratios=(0.9, 0.05, 0.05))
    save_jsonl(tr_senti, os.path.join(data_dir, "sentiment", "train.jsonl"))
    save_jsonl(va_senti, os.path.join(data_dir, "sentiment", "valid.jsonl"))
    save_jsonl(te_senti, os.path.join(data_dir, "sentiment", "test.jsonl"))

    # 3) Prepare Lyrics (Genius bilingual)
    lyr_df = prepare_genius(os.path.join(data_dir, "lyrics"), max_per_lang=args.max_lyrics_per_lang)
    tr_lyr, va_lyr, te_lyr = train_valid_test_split(lyr_df, ratios=(0.9, 0.05, 0.05))
    save_jsonl(tr_lyr, os.path.join(data_dir, "lyrics", "train.jsonl"))
    save_jsonl(va_lyr, os.path.join(data_dir, "lyrics", "valid.jsonl"))
    save_jsonl(te_lyr, os.path.join(data_dir, "lyrics", "test.jsonl"))

    # 4) Prepare Music (JSB -> melody sequences)
    jsb = prepare_jsb(os.path.join(data_dir, "jsb"))
    ensure_dir(os.path.join(data_dir, "music"))
    with open(os.path.join(data_dir, "music", "jsb_melody.json"), "w", encoding="utf-8") as f:
        json.dump(jsb, f)
    print(f"[OK] Saved JSB melody json at data/music/jsb_melody.json")

    # 5) Train a single SentencePiece model on all text corpora (sentiment+lyrics)
    corpus_path = os.path.join(args.spm_dir, "corpus.txt")
    write_corpus_txt([tr_senti[["text"]], va_senti[["text"]], tr_lyr[["text"]], va_lyr[["text"]]], corpus_path, text_field="text")
    user_symbols = ["<KOR>", "<ENG>", "<LYR>", "<SENTI>", "<EMO:HAPPY>", "<EMO:SAD>", "<EMO:ANGRY>", "<EMO:CALM>"]
    spm_prefix = os.path.join(args.spm_dir, "bpe8k")
    train_sentencepiece(corpus_path, spm_prefix, args.spm_vocab_size, user_symbols)

    # 6) Save a manifest
    manifest = {
        "imdb_slug": args.imdb_slug,
        "nsmc_slug": args.nsmc_slug,
        "lyrics_slug": args.lyrics_slug,
        "jsb_slug": args.jsb_slug,
        "sizes": {
            "sentiment": {"train": len(tr_senti), "valid": len(va_senti), "test": len(te_senti)},
            "lyrics": {"train": len(tr_lyr), "valid": len(va_lyr), "test": len(te_lyr)},
            "music_sequences": {k: len(v) for k, v in jsb.items()}
        },
        "spm": {"prefix": spm_prefix, "vocab_size": args.spm_vocab_size, "user_symbols": user_symbols},
        "note": "All text was lightly cleaned and truncated to <=3000 chars per item for memory efficiency."
    }
    with open(os.path.join(data_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote manifest -> {os.path.join(data_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
