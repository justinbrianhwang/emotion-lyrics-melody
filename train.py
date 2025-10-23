
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

학습/추론 스크립트
- 감정 분석: LSTM 분류기 vs Transformer Encoder 분류기
- 가사 생성: LSTM LM vs GPT-like Transformer LM
- 음악 생성(멜로디): Note-LSTM vs Note-Transformer
- end-to-end demo: 사용자 "현재 상황" -> 감정 예측 -> 감정 맞춤 가사 생성 -> 멜로디 생성 -> WAV 렌더링

필요 패키지: torch, pandas, numpy, tqdm, sentencepiece, scikit-learn
선택: pretty_midi (MIDI 저장), but 여기서는 간단한 오실레이터로 WAV 생성

작성: GPT-5 Pro
"""
import argparse
import os
import io
import json
import math
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm
from sklearn.metrics import accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Utils
# ---------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def save_checkpoint(path: str, payload: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)
    print(f"[OK] Saved checkpoint: {path} (size ~ {os.path.getsize(path)/1e6:.1f} MB)")


def has_korean(text: str) -> bool:
    return bool(re.search(r"[\u3131-\u318E\uAC00-\uD7A3]", text))


# ---------------------------
# Tokenizer
# ---------------------------
class SPTokenizer:
    def __init__(self, spm_model_path: str, special_tokens: List[str] = None):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.special_tokens = special_tokens or []
        # map specials
        self.special_to_id = {tok: self.sp.piece_to_id(tok) for tok in self.special_tokens if tok in self.sp}
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, add_bos=False, add_eos=False, max_len: int = None) -> List[int]:
        ids = self.sp.encode_as_ids(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        # strip specials
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode_ids(ids)


# ---------------------------
# Datasets for text
# ---------------------------
class TextClassificationDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: SPTokenizer, max_len: int = 256, add_lang_token=True):
        self.samples = load_jsonl(jsonl_path)
        self.tok = tokenizer
        self.max_len = max_len
        self.add_lang_token = add_lang_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        if self.add_lang_token:
            lang_tok = "<KOR>" if item.get("lang") == "kor" or has_korean(text) else "<ENG>"
            text = f"{lang_tok} <SENTI> " + text
        ids = self.tok.encode(text, add_bos=True, add_eos=True, max_len=self.max_len)
        label = int(item["label"])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def pad_sequence(seqs: List[torch.Tensor], pad_value=0):
    max_len = max(len(x) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, x in enumerate(seqs):
        out[i, :len(x)] = x
        attn[i, :len(x)] = 1
    return out, attn


def collate_classification(batch):
    xs, ys = zip(*batch)
    xpad, attn = pad_sequence(xs, pad_value=0)
    y = torch.stack(ys)
    return xpad, attn, y


class TextLMdataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: SPTokenizer, max_len: int = 256, add_lang_token=True):
        self.samples = load_jsonl(jsonl_path)
        self.tok = tokenizer
        self.max_len = max_len
        self.add_lang_token = add_lang_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        lang_tok = "<KOR>" if item.get("lang") == "kor" or has_korean(text) else "<ENG>"
        prefix = f"{lang_tok} <LYR> "
        if self.add_lang_token:
            text = prefix + text
        ids = self.tok.encode(text, add_bos=True, add_eos=True, max_len=self.max_len)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_lm(batch):
    xs, ys = zip(*batch)
    xpad, _ = pad_sequence(xs, pad_value=0)
    ypad, _ = pad_sequence(ys, pad_value=0)
    return xpad, ypad


# ---------------------------
# Models
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, n_layers=1, bidirectional=True, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model//2 if bidirectional else d_model, num_layers=n_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(d_model, num_classes) if bidirectional else nn.Linear(d_model, num_classes)

    def forward(self, x, attn_mask):
        emb = self.emb(x)  # [B, T, D]
        out, _ = self.lstm(emb)  # [B, T, D]
        # Masked mean pooling
        mask = attn_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        logits = self.fc(pooled)
        return logits


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, n_heads=4, n_layers=4, dim_ff=512, num_classes=2, pdrop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, 2048, d_model))  # max 2048
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, dropout=pdrop, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, attn_mask):
        b, t = x.size()
        pos = self.pos[:, :t, :]
        h = self.emb(x) + pos
        # src_key_padding_mask: True where to mask
        src_key_padding_mask = ~attn_mask
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        # CLS-less: mean pooling
        mask = attn_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        logits = self.fc(pooled)
        return logits


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, d_model=384, n_layers=2, pdrop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop = nn.Dropout(pdrop)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.drop(self.emb(x))
        h, _ = self.lstm(h)
        h = self.ln(h)
        return self.head(h)


class GPTMini(nn.Module):
    def __init__(self, vocab_size: int, d_model=384, n_layers=4, n_heads=6, dim_ff=1024, pdrop=0.1, max_ctx=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_ctx, d_model))
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
                                           dropout=pdrop, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, t = x.shape
        pos = self.pos[:, :t, :]
        h = self.emb(x) + pos
        # causal mask
        causal = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        h = self.dec(h, h, tgt_mask=causal)  # decoder-only trick
        h = self.ln(h)
        return self.head(h)


# Music models (note sequences)
class NoteLSTM(nn.Module):
    def __init__(self, n_tokens: int, d_model=256, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.lstm(h)
        return self.head(h)


class NoteTransformer(nn.Module):
    def __init__(self, n_tokens: int, d_model=256, n_layers=4, n_heads=4, dim_ff=512, max_ctx=2048):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_ctx, d_model))
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_ff, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        b, t = x.shape
        pos = self.pos[:, :t, :]
        h = self.emb(x) + pos
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        h = self.dec(h, h, tgt_mask=mask)
        return self.head(h)


# ---------------------------
# Datasets for music (note integers)
# ---------------------------
class NoteDataset(Dataset):
    def __init__(self, json_path: str, split: str = "train", max_len: int = 256):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.seqs = data.get(split, [])
        # prepend BOS(1) and append EOS(2); reserve 0 as "rest/pad"
        self.BOS = 1
        self.EOS = 2
        self.max_len = max_len

        # map MIDI/rest to compact token space
        # rest=0 -> 0, MIDI 21..108 -> 21..108 (we'll cap to 0..128)
        self.vocab_size = 129  # 0..128 (inclusive) use 0 pad/rest, 1 BOS, 2 EOS, 3..128 for MIDI
        # We will remap MIDI 21..108 -> 3..90, but keep it simple and allow sparse ids.

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        raw = self.seqs[idx]
        # truncate and add BOS/EOS
        seq = [self.BOS] + [int(x) if 0 <= int(x) <= 128 else 0 for x in raw][: self.max_len-2] + [self.EOS]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def collate_note(batch):
    xs, ys = zip(*batch)
    # simple pad
    max_len = max(len(x) for x in xs)
    xpad = torch.zeros((len(xs), max_len), dtype=torch.long)
    ypad = torch.zeros((len(xs), max_len), dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xpad[i, :len(x)] = x
        ypad[i, :len(y)] = y
    return xpad, ypad


# ---------------------------
# Training loops
# ---------------------------
def train_text_classifier(model, train_loader, valid_loader, epochs=3, lr=2e-4, wd=0.01, grad_clip=1.0):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = {"acc": 0.0, "state": None}
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for xb, attn, y in tqdm(train_loader, desc=f"[Cls][Train] ep{ep}"):
            xb, attn, y = xb.to(DEVICE), attn.to(DEVICE), y.to(DEVICE)
            logits = model(xb, attn)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())

        # Eval
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, attn, y in tqdm(valid_loader, desc=f"[Cls][Valid] ep{ep}"):
                xb, attn = xb.to(DEVICE), attn.to(DEVICE)
                logits = model(xb, attn)
                p = logits.argmax(-1).cpu().tolist()
                preds.extend(p)
                gts.extend(y.tolist())
        acc = accuracy_score(gts, preds)
        print(f"Epoch {ep}: train loss {np.mean(losses):.4f}, valid acc {acc:.4f}")
        if acc > best["acc"]:
            best["acc"] = acc
            best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
    return best


def train_lm(model, train_loader, valid_loader, epochs=3, lr=2e-4, wd=0.01, grad_clip=1.0):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = {"ppl": float("inf"), "state": None}
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"[LM][Train] ep{ep}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())
        # Eval PPL
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for xb, yb in tqdm(valid_loader, desc=f"[LM][Valid] ep{ep}"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=0)
                eval_losses.append(loss.item())
        ppl = math.exp(np.mean(eval_losses))
        print(f"Epoch {ep}: train loss {np.mean(losses):.4f}, valid ppl {ppl:.2f}")
        if ppl < best["ppl"]:
            best["ppl"] = ppl
            best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
    return best


# ---------------------------
# Generation helpers
# ---------------------------
def top_k_top_p_filter(logits, top_k=50, top_p=0.9):
    # logits: [V]
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    if top_k is not None and top_k > 0:
        sorted_probs[top_k:] = 0
    if top_p is not None and 0.0 < top_p < 1.0:
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff_idx = torch.where(cutoff)[0]
        if len(cutoff_idx) > 0:
            first = cutoff_idx[0]
            sorted_probs[first+1:] = 0
    sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-8)
    idx = torch.multinomial(sorted_probs, num_samples=1).item()
    return sorted_idx[idx].item()


@torch.no_grad()
def generate_text(model, tok: SPTokenizer, prompt: str, max_new_tokens=200, temperature=1.0, top_k=50, top_p=0.9):
    model.to(DEVICE).eval()
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        logits = logits / max(temperature, 1e-4)
        next_id = top_k_top_p_filter(logits.squeeze(0), top_k=top_k, top_p=top_p)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        if next_id == tok.eos_id:
            break
    return tok.decode(ids[0].tolist())


@torch.no_grad()
def generate_notes(model, start_token=1, max_steps=256, temperature=1.0, top_k=8, top_p=0.9):
    model.to(DEVICE).eval()
    ids = torch.tensor([[start_token]], dtype=torch.long, device=DEVICE)
    out = []
    for _ in range(max_steps):
        logits = model(ids)[:, -1, :]
        logits = logits / max(temperature, 1e-4)
        # sample with top-k/p
        probs = F.softmax(logits.squeeze(0), dim=-1)
        if top_k is not None and top_k > 0:
            vals, inds = torch.topk(probs, k=min(top_k, probs.numel()))
            mask = torch.zeros_like(probs)
            mask.scatter_(0, inds, vals)
            probs = mask
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff_idx = torch.where(cutoff)[0]
            if len(cutoff_idx) > 0:
                first = cutoff_idx[0]
                sorted_probs[first+1:] = 0
            # restore order
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_idx, sorted_probs)
        probs = probs / (probs.sum() + 1e-8)
        nxt = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        if nxt == 2:  # EOS
            break
        if nxt != 0:
            out.append(nxt)
    return out


def render_wav(notes: List[int], bpm=90, sr=22050, outfile="song.wav"):
    """
    간단한 신디사이저로 멜로디를 WAV로 렌더.
    - rest(0)는 무음, MIDI 21..108은 사인파 기반 오실레이터
    - 각 step 길이는 16분음표 기준으로 1 step
    """
    import numpy as np
    from scipy.io import wavfile

    sec_per_beat = 60.0 / bpm
    step_dur = sec_per_beat / 4.0  # 16th note

    # ADSR
    def envelope(n_samples, sr):
        a = int(0.01 * sr)
        d = int(0.05 * sr)
        s = max(0, n_samples - a - d - int(0.05 * sr))
        r = int(0.05 * sr)
        env = np.zeros(n_samples, dtype=np.float32)
        if n_samples <= a + d + r:
            env[:n_samples] = np.linspace(0, 1, n_samples)
            return env
        env[:a] = np.linspace(0, 1, a)
        env[a:a+d] = np.linspace(1, 0.7, d)
        env[a+d:a+d+s] = 0.7
        env[a+d+s:a+d+s+r] = np.linspace(0.7, 0, r)
        return env

    def midi_to_freq(m):
        return 440.0 * (2 ** ((m - 69) / 12.0))

    audio = np.zeros(int(len(notes) * step_dur * sr) + sr, dtype=np.float32)
    t = 0
    for note in notes:
        n_samples = int(step_dur * sr)
        if note == 0:
            t += n_samples
            continue
        f = midi_to_freq(note)
        # 2-osc (sine + slight detune)
        x = np.sin(2*np.pi*f*np.arange(n_samples)/sr) + 0.3*np.sin(2*np.pi*(f*1.01)*np.arange(n_samples)/sr)
        env = envelope(n_samples, sr)
        frame = (x * env).astype(np.float32) * 0.2
        audio[t:t+n_samples] += frame
        t += n_samples

    # normalize
    mx = np.max(np.abs(audio)) + 1e-8
    audio = (audio / mx * 0.9).astype(np.float32)
    wavfile.write(outfile, sr, audio)
    return outfile


# ---------------------------
# Packing helpers
# ---------------------------
def pack_text_data_for_checkpoint(train_jsonl: str, valid_jsonl: str, max_chars: int = 2_000_000) -> Dict[str, Any]:
    """
    모델 체크포인트 안에 '학습에 실제 사용된 텍스트'를 같이 저장해
    추가 다운로드 없이도 재현/데모가 가능하도록 함.
    max_chars로 너무 커지는 것을 방지.
    """
    def read_trim(path):
        buf = []
        total = 0
        for obj in load_jsonl(path):
            s = obj.get("text", "")
            if not s:
                continue
            if total + len(s) > max_chars:
                # cut last chunk
                s = s[:max(0, max_chars - total)]
                buf.append({"text": s, "lang": obj.get("lang"), "label": obj.get("label")})
                break
            buf.append({"text": s, "lang": obj.get("lang"), "label": obj.get("label")})
            total += len(s)
        return buf
    return {
        "train_subset": read_trim(train_jsonl),
        "valid_subset": read_trim(valid_jsonl)
    }


def pack_music_data_for_checkpoint(json_path: str, split="train", max_len=50000) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seqs = data.get(split, [])
    # flatten to single list with cut
    flat = []
    for s in seqs:
        flat.extend([int(x) for x in s][:256])
        if len(flat) > max_len:
            break
    return {"melody_subset": flat[:max_len]}


# ---------------------------
# CLI Commands
# ---------------------------
def cmd_train_sentiment(args):
    # Tokenizer
    tok = SPTokenizer(args.spm_model, special_tokens=["<KOR>", "<ENG>", "<LYR>", "<SENTI>",
                                                      "<EMO:HAPPY>", "<EMO:SAD>", "<EMO:ANGRY>", "<EMO:CALM>"])
    # Datasets
    tr = TextClassificationDataset(os.path.join(args.data_dir, "sentiment", "train.jsonl"), tok, max_len=args.max_len)
    va = TextClassificationDataset(os.path.join(args.data_dir, "sentiment", "valid.jsonl"), tok, max_len=args.max_len)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_classification, num_workers=2)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_classification, num_workers=2)

    if args.model_type == "lstm":
        model = LSTMClassifier(tok.vocab_size, d_model=args.d_model, n_layers=args.n_layers)
    else:
        model = TransformerClassifier(tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                                      n_layers=args.n_layers, dim_ff=args.dim_ff)

    best = train_text_classifier(model, tr_loader, va_loader, epochs=args.epochs, lr=args.lr, wd=args.wd)
    # Pack & save
    pack = pack_text_data_for_checkpoint(os.path.join(args.data_dir, "sentiment", "train.jsonl"),
                                         os.path.join(args.data_dir, "sentiment", "valid.jsonl"),
                                         max_chars=args.pack_max_chars)
    payload = {
        "model_type": f"sentiment_{args.model_type}",
        "state_dict": best["state"],
        "vocab_size": tok.vocab_size,
        "spm_model_bytes": open(args.spm_model, "rb").read(),
        "special_tokens": tok.special_tokens,
        "train_metrics": {"best_valid_acc": best["acc"]},
        "packed_data": pack,
        "config": vars(args)
    }
    out = os.path.join(args.out_dir, f"sentiment_{args.model_type}.pth")
    save_checkpoint(out, payload)


def cmd_train_lyrics(args):
    tok = SPTokenizer(args.spm_model, special_tokens=["<KOR>", "<ENG>", "<LYR>", "<SENTI>",
                                                      "<EMO:HAPPY>", "<EMO:SAD>", "<EMO:ANGRY>", "<EMO:CALM>"])
    tr = TextLMdataset(os.path.join(args.data_dir, "lyrics", "train.jsonl"), tok, max_len=args.max_len)
    va = TextLMdataset(os.path.join(args.data_dir, "lyrics", "valid.jsonl"), tok, max_len=args.max_len)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_lm, num_workers=2)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_lm, num_workers=2)

    if args.model_type == "lstm":
        model = LSTMLM(tok.vocab_size, d_model=args.d_model, n_layers=args.n_layers)
    else:
        model = GPTMini(tok.vocab_size, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, dim_ff=args.dim_ff)

    best = train_lm(model, tr_loader, va_loader, epochs=args.epochs, lr=args.lr, wd=args.wd)
    pack = pack_text_data_for_checkpoint(os.path.join(args.data_dir, "lyrics", "train.jsonl"),
                                         os.path.join(args.data_dir, "lyrics", "valid.jsonl"),
                                         max_chars=args.pack_max_chars)
    payload = {
        "model_type": f"lyrics_{args.model_type}",
        "state_dict": best["state"],
        "vocab_size": tok.vocab_size,
        "spm_model_bytes": open(args.spm_model, "rb").read(),
        "special_tokens": tok.special_tokens,
        "train_metrics": {"best_valid_ppl": best["ppl"]},
        "packed_data": pack,
        "config": vars(args)
    }
    out = os.path.join(args.out_dir, f"lyrics_{args.model_type}.pth")
    save_checkpoint(out, payload)


def cmd_train_music(args):
    json_path = os.path.join(args.data_dir, "music", "jsb_melody.json")
    tr = NoteDataset(json_path, split="train", max_len=args.max_len)
    va = NoteDataset(json_path, split="valid", max_len=args.max_len)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_note, num_workers=2)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_note, num_workers=2)

    n_tokens = tr.vocab_size
    if args.model_type == "lstm":
        model = NoteLSTM(n_tokens, d_model=args.d_model, n_layers=args.n_layers)
    else:
        model = NoteTransformer(n_tokens, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, dim_ff=args.dim_ff, max_ctx=args.max_len)

    best = train_lm(model, tr_loader, va_loader, epochs=args.epochs, lr=args.lr, wd=args.wd)

    pack = pack_music_data_for_checkpoint(json_path, split="train", max_len=args.pack_music_len)
    payload = {
        "model_type": f"music_{args.model_type}",
        "state_dict": best["state"],
        "n_tokens": n_tokens,
        "train_metrics": {"best_valid_ppl": best["ppl"]},
        "packed_data": pack,
        "config": vars(args)
    }
    out = os.path.join(args.out_dir, f"music_{args.model_type}.pth")
    save_checkpoint(out, payload)


def load_sp_from_bytes(spm_bytes: bytes, tmp_path: str = "./_tmp_spm.model"):
    with open(tmp_path, "wb") as f:
        f.write(spm_bytes)
    tok = SPTokenizer(tmp_path, special_tokens=["<KOR>", "<ENG>", "<LYR>", "<SENTI>",
                                                "<EMO:HAPPY>", "<EMO:SAD>", "<EMO:ANGRY>", "<EMO:CALM>"])
    return tok, tmp_path


@torch.no_grad()
def cmd_infer_pipeline(args):
    """
    End-to-end:
    1) 감정 분석 (두 모델로 각각 추론 비교)
    2) 예측 감정 텍스트 출력
    3) 가사 생성 (LSTM vs Transformer 비교)
    4) 음악 생성 (LSTM vs Transformer 비교)
    5) WAV 파일 저장
    """
    # 1) Load checkpoints
    s_lstm = torch.load(args.sentiment_lstm, map_location="cpu")
    s_tr = torch.load(args.sentiment_tr, map_location="cpu")
    ly_lstm = torch.load(args.lyrics_lstm, map_location="cpu")
    ly_tr = torch.load(args.lyrics_tr, map_location="cpu")
    mu_lstm = torch.load(args.music_lstm, map_location="cpu")
    mu_tr = torch.load(args.music_tr, map_location="cpu")

    # Tokenizer from any lyrics/senti ckpt
    tok, tmp_path = load_sp_from_bytes(ly_tr["spm_model_bytes"], tmp_path="./_tmp_spm.model")

    # Rebuild models
    def build_sent(model_payload):
        vocab_size = model_payload.get("vocab_size", tok.vocab_size)
        if model_payload["model_type"].endswith("lstm"):
            m = LSTMClassifier(vocab_size, d_model=256)
        else:
            m = TransformerClassifier(vocab_size, d_model=256, n_heads=4, n_layers=4, dim_ff=512)
        m.load_state_dict({k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in model_payload["state_dict"].items()})
        return m.to(DEVICE).eval()

    def build_lyric(model_payload):
        vocab_size = model_payload.get("vocab_size", tok.vocab_size)
        if model_payload["model_type"].endswith("lstm"):
            m = LSTMLM(vocab_size, d_model=384, n_layers=2)
        else:
            m = GPTMini(vocab_size, d_model=384, n_layers=4, n_heads=6, dim_ff=1024)
        m.load_state_dict({k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in model_payload["state_dict"].items()})
        return m.to(DEVICE).eval()

    def build_music(model_payload, n_tokens=129):
        if model_payload["model_type"].endswith("lstm"):
            m = NoteLSTM(n_tokens, d_model=256, n_layers=2)
        else:
            m = NoteTransformer(n_tokens, d_model=256, n_layers=4, n_heads=4, dim_ff=512, max_ctx=256)
        m.load_state_dict({k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in model_payload["state_dict"].items()})
        return m.to(DEVICE).eval()

    s_m_lstm = build_sent(s_lstm)
    s_m_tr = build_sent(s_tr)
    ly_m_lstm = build_lyric(ly_lstm)
    ly_m_tr = build_lyric(ly_tr)
    mu_m_lstm = build_music(mu_lstm, n_tokens=mu_lstm.get("n_tokens", 129))
    mu_m_tr = build_music(mu_tr, n_tokens=mu_tr.get("n_tokens", 129))

    # 2) Sentiment prediction
    text = args.input_text
    lang_tok = "<KOR>" if has_korean(text) else "<ENG>"
    senti_prompt = f"{lang_tok} <SENTI> " + text
    x_ids = torch.tensor([tok.encode(senti_prompt, add_bos=True, add_eos=True, max_len=args.max_len)], dtype=torch.long, device=DEVICE)
    attn = torch.ones_like(x_ids).bool()

    logits_lstm = s_m_lstm(x_ids, attn)
    logits_tr = s_m_tr(x_ids, attn)
    prob_lstm = F.softmax(logits_lstm, dim=-1).squeeze(0).cpu().numpy().tolist()
    prob_tr = F.softmax(logits_tr, dim=-1).squeeze(0).cpu().numpy().tolist()
    pred_lstm = int(np.argmax(prob_lstm))
    pred_tr = int(np.argmax(prob_tr))

    label_map = {0: "NEGATIVE(부정)", 1: "POSITIVE(긍정)"}
    print("[Sentiment] LSTM:", label_map[pred_lstm], prob_lstm)
    print("[Sentiment] TR  :", label_map[pred_tr], prob_tr)

    # 3) Map to coarse emotion for conditioning
    # 간단 매핑: positive -> HAPPY/CALM, negative -> SAD/ANGRY (문장 길이로 튜닝)
    emo = "<EMO:HAPPY>" if (pred_tr == 1 or pred_lstm == 1) else "<EMO:SAD>"
    print("[Emotion chosen for generation]:", emo)

    # 4) Lyrics generation
    prefer_lang = "KOR" if has_korean(text) else "ENG"
    lyr_prompt = f"<{prefer_lang}> <LYR> {emo} "
    out_lstm = generate_text(ly_m_lstm, tok, lyr_prompt, max_new_tokens=args.lyrics_tokens, temperature=1.0, top_k=50, top_p=0.9)
    out_tr = generate_text(ly_m_tr, tok, lyr_prompt, max_new_tokens=args.lyrics_tokens, temperature=0.9, top_k=50, top_p=0.9)
    print("\n=== Lyrics (LSTM) ===\n", out_lstm)
    print("\n=== Lyrics (Transformer) ===\n", out_tr)

    # 5) Music generation
    notes_lstm = generate_notes(mu_m_lstm, max_steps=args.music_steps, temperature=1.0, top_k=8, top_p=0.9)
    notes_tr = generate_notes(mu_m_tr, max_steps=args.music_steps, temperature=0.9, top_k=8, top_p=0.9)

    # adjust tempo by emotion
    bpm = 100 if emo == "<EMO:HAPPY>" else 72
    out_wav_lstm = render_wav(notes_lstm, bpm=bpm, outfile=os.path.join(args.out_dir, "song_lstm.wav"))
    out_wav_tr = render_wav(notes_tr, bpm=bpm, outfile=os.path.join(args.out_dir, "song_tr.wav"))
    print(f"[OK] Rendered audio ->\n  LSTM: {out_wav_lstm}\n  TR  : {out_wav_tr}")


def main():
    parser = argparse.ArgumentParser(description="Train & Inference for LSTM vs Transformer (Sentiment, Lyrics, Music)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared defaults
    def add_shared(p):
        p.add_argument("--data_dir", type=str, default="./data")
        p.add_argument("--out_dir", type=str, default="./outputs")
        p.add_argument("--max_len", type=int, default=256)
        p.add_argument("--batch_size", type=int, default=32)
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--wd", type=float, default=0.01)
        p.add_argument("--d_model", type=int, default=256)
        p.add_argument("--n_layers", type=int, default=4)
        p.add_argument("--n_heads", type=int, default=4)
        p.add_argument("--dim_ff", type=int, default=512)
        p.add_argument("--spm_model", type=str, default="./spm/bpe8k.model")
        p.add_argument("--pack_max_chars", type=int, default=2_000_000)
        p.add_argument("--pack_music_len", type=int, default=50000)

    # train_sentiment
    p1 = sub.add_parser("train_sentiment")
    add_shared(p1)
    p1.add_argument("--model_type", type=str, choices=["lstm", "transformer"], default="transformer")
    p1.set_defaults(func=cmd_train_sentiment)

    # train_lyrics
    p2 = sub.add_parser("train_lyrics")
    add_shared(p2)
    p2.add_argument("--model_type", type=str, choices=["lstm", "transformer"], default="transformer")
    p2.set_defaults(func=cmd_train_lyrics)

    # train_music
    p3 = sub.add_parser("train_music")
    add_shared(p3)
    p3.add_argument("--model_type", type=str, choices=["lstm", "transformer"], default="transformer")
    p3.set_defaults(func=cmd_train_music)

    # infer pipeline
    p4 = sub.add_parser("infer")
    p4.add_argument("--input_text", type=str, required=True, help="현재 상황/감정에 대한 문장 입력")
    p4.add_argument("--sentiment_lstm", type=str, required=True)
    p4.add_argument("--sentiment_tr", type=str, required=True)
    p4.add_argument("--lyrics_lstm", type=str, required=True)
    p4.add_argument("--lyrics_tr", type=str, required=True)
    p4.add_argument("--music_lstm", type=str, required=True)
    p4.add_argument("--music_tr", type=str, required=True)
    p4.add_argument("--spm_model", type=str, default="./spm/bpe8k.model")
    p4.add_argument("--out_dir", type=str, default="./outputs")
    p4.add_argument("--max_len", type=int, default=256)
    p4.add_argument("--lyrics_tokens", type=int, default=120)
    p4.add_argument("--music_steps", type=int, default=256)
    p4.set_defaults(func=cmd_infer_pipeline)

    args = parser.parse_args()
    ensure_dir("./outputs")
    args.func(args)


if __name__ == "__main__":
    main()
