"""
Evaluate Whisper (fine-tuned on original clean NSS)
on augmented NSS dataset (clean + MUSAN noise).

Usage:
    python3 eval_org_augmented.py
    python3 eval_org_augmented.py \
        --checkpoint train_whisp_ga_large_v3_nss_original/best_model \
        --augmented_dir augmented_ga_nss_musan \
        --output_dir eval_results_org
"""

import io
import os
import re
import csv
import time
import argparse
import json
import torch
import numpy as np
import soundfile as sf
import evaluate as hf_evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT  = "train_whisp_ga_large_v3_nss_original/best_model"
DEFAULT_AUG_DIR     = "augmented_ga_nss_musan"
DEFAULT_OUTPUT_DIR  = "eval_results_org"
BASE_MODEL          = "openai/whisper-large-v3"
LANGUAGE            = "yo"
TASK                = "transcribe"
SR                  = 16000
EVAL_SPLITS         = ["validation", "test"]
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLS = [
    "split", "example_idx", "speaker_id",
    "recording_environment", "recording_device",
    "snr_db", "snr_label", "noise_type",
    "audio_length", "transcript_length",
    "reference", "prediction",
    "reference_normalized", "prediction_normalized",
    "wer", "cer", "correct",
    "inference_time_s", "rtf",
]

SUMMARY_COLS = [
    "split", "group_type", "group_value",
    "n_examples", "overall_wer", "overall_cer",
    "avg_wer", "avg_cer",
]


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--augmented_dir", type=str, default=DEFAULT_AUG_DIR)
    p.add_argument("--output_dir",    type=str, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_wav(path: str) -> np.ndarray:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_processor(checkpoint: str):
    """
    Load processor from BASE_MODEL (avoids tokenizer version bug).
    Load weights from checkpoint.
    Use the exact same generation config that works on clean NSS.
    """
    print(f"\nLoading processor from: {BASE_MODEL}")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language=LANGUAGE, task=TASK
    )

    print(f"Loading model from   : {checkpoint}")
    model = WhisperForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16
    ).cuda()
    model.eval()

    # ── Exact config that works on clean NSS (WER=0.493) ─────────────────────
    # Do NOT set suppress_tokens — leave whatever the checkpoint saved
    # Do NOT set forced_decoder_ids on config — only on generation_config
    model.config.forced_decoder_ids           = None
    model.config.use_cache                    = True
    model.generation_config.language          = LANGUAGE
    model.generation_config.task              = TASK
    model.generation_config.return_timestamps = False
    model.generation_config.forced_decoder_ids = None

    # ── Quick sanity check on one dummy input ─────────────────────────────────
    print("\nSanity check — generating from silence...")
    dummy = np.zeros(SR * 2, dtype=np.float32)  # 2s silence
    feats = processor(dummy, sampling_rate=SR, return_tensors="pt"
                      ).input_features.to(torch.bfloat16).cuda()
    with torch.no_grad():
        ids = model.generate(feats)
    decoded = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"  Dummy output: '{decoded}' (empty is ok for silence)")

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
def transcribe_one(model, processor, wav_path: str):
    """
    Load one WAV, run inference, return (prediction, inference_time_s).
    Falls back to raw decode if skip_special_tokens gives empty string.
    """
    array = load_wav(wav_path)
    feats = processor(
        array, sampling_rate=SR, return_tensors="pt"
    ).input_features.to(torch.bfloat16).cuda()

    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(feats)
    t1 = time.perf_counter()

    pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(pred)
    # If empty — decode without skipping and strip manually
    if not pred.strip():
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        # Remove known special tokens manually
        for tok in ['<|startoftranscript|>', f'<|{LANGUAGE}|>', '<|transcribe|>',
                    '<|notimestamps|>', '<|endoftext|>', '<|nospeech|>']:
            raw = raw.replace(tok, '')
        pred = raw.strip()
        print(pred)

    return pred, round(t1 - t0, 4)


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_split(split_name, aug_dir, model, processor,
                   wer_metric, cer_metric):
    """
    Evaluate all rows (clean + musan) for one split.
    Reads from metadata.csv, loads WAVs from audio/{split}/.
    """
    csv_path = os.path.join(aug_dir, "metadata.csv")
    split_rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split_name:
                split_rows.append(row)

    total     = len(split_rows)
    n_clean   = sum(1 for r in split_rows if r.get("noise_type") == "clean")
    n_musan   = sum(1 for r in split_rows if r.get("noise_type") == "musan")
    print(f"\n[{split_name.upper()}]  {total:,} rows  "
          f"({n_clean:,} clean + {n_musan:,} musan)")

    rows      = []
    all_refs  = []
    all_preds = []
    n_empty   = 0
    n_skip    = 0

    for i, meta in enumerate(split_rows):
        if i % 500 == 0:
            print(f"  [{i:>6}/{total}] empty={n_empty} skip={n_skip} ...")

        wav_path = os.path.join(
            aug_dir, "audio", meta["split"], meta["audio_filename"]
        )

        try:
            pred, inf_time = transcribe_one(model, processor, wav_path)
            ref            = meta.get("transcription", "")

            pred_norm = normalize(pred)
            ref_norm  = normalize(ref)

            if not pred_norm:
                n_empty += 1
                pred_norm = ""   # keep empty — do not skip row

            wer_val = wer_metric.compute(
                predictions=[pred_norm], references=[ref_norm]
            ) if ref_norm else 0.0
            cer_val = cer_metric.compute(
                predictions=[pred_norm], references=[ref_norm]
            ) if ref_norm else 0.0

            snr_db    = float(meta.get("snr_db", 999))
            audio_len = float(meta.get("audio_length") or 0)
            rtf       = round(inf_time / audio_len, 4) if audio_len > 0 else ""

            all_refs.append(ref_norm)
            all_preds.append(pred_norm)

            rows.append({
                "split":                  split_name,
                "example_idx":            i,
                "speaker_id":             meta.get("speaker_id", ""),
                "recording_environment":  meta.get("recording_environment", ""),
                "recording_device":       meta.get("recording_device", ""),
                "snr_db":                 snr_db,
                "snr_label":              meta.get("snr_label", "clean"),
                "noise_type":             meta.get("noise_type", "clean"),
                "audio_length":           meta.get("audio_length", ""),
                "transcript_length":      meta.get("transcript_length", ""),
                "reference":              ref,
                "prediction":             pred,
                "reference_normalized":   ref_norm,
                "prediction_normalized":  pred_norm,
                "wer":                    round(wer_val, 4),
                "cer":                    round(cer_val, 4),
                "correct":                1 if wer_val == 0.0 else 0,
                "inference_time_s":       inf_time,
                "rtf":                    rtf,
            })

        except Exception as e:
            n_skip += 1
            print(f"  SKIP {i}: {e}")

    overall_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    overall_cer = cer_metric.compute(predictions=all_preds, references=all_refs)

    print(f"\n  [{split_name.upper()}] Overall WER={overall_wer:.4f}  "
          f"CER={overall_cer:.4f}")
    print(f"  Empty predictions : {n_empty}")
    print(f"  Skipped (error)   : {n_skip}")

    # Sample predictions
    print(f"\n  Sample predictions (first clean example):")
    for r in rows:
        if r["noise_type"] == "clean":
            print(f"    REF : {r['reference_normalized'][:80]}")
            print(f"    PRED: {r['prediction_normalized'][:80]}")
            print(f"    WER : {r['wer']}")
            break

    return rows, overall_wer, overall_cer


# ─────────────────────────────────────────────────────────────────────────────
def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_COLS})
    print(f"  Saved: {path}  ({len(rows):,} rows)")


def save_summary(rows, split_name, output_dir):
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    def empty():
        return {"refs": [], "preds": [], "wers": [], "cers": []}

    spk_data = defaultdict(empty)
    snr_data = defaultdict(empty)
    env_data = defaultdict(empty)

    for row in rows:
        for key, store in [
            (str(row.get("speaker_id", "unknown")), spk_data),
            (str(row.get("snr_label", "clean")),    snr_data),
            (str(row.get("recording_environment", "unknown")), env_data),
        ]:
            store[key]["refs"].append(row["reference_normalized"])
            store[key]["preds"].append(row["prediction_normalized"])
            store[key]["wers"].append(float(row["wer"]))
            store[key]["cers"].append(float(row["cer"]))

    summary_rows = []

    def add_group(group_type, group_value, d):
        if not d["refs"]:
            return
        summary_rows.append({
            "split":       split_name,
            "group_type":  group_type,
            "group_value": group_value,
            "n_examples":  len(d["refs"]),
            "overall_wer": round(wer_metric.compute(
                predictions=d["preds"], references=d["refs"]), 4),
            "overall_cer": round(cer_metric.compute(
                predictions=d["preds"], references=d["refs"]), 4),
            "avg_wer":     round(np.mean(d["wers"]), 4),
            "avg_cer":     round(np.mean(d["cers"]), 4),
        })

    for spk in sorted(spk_data):
        add_group("speaker", spk, spk_data[spk])

    for snr in ["clean", "-10dB", "-5dB", "+0dB", "+5dB", "+10dB", "+20dB"]:
        if snr in snr_data:
            add_group("snr", snr, snr_data[snr])

    for env in sorted(env_data):
        add_group("environment", env, env_data[env])

    path = os.path.join(output_dir, f"summary_{split_name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Saved: {path}  ({len(summary_rows)} groups)")

    print(f"\n  [{split_name.upper()}] Summary by group:")
    print(f"  {'Group':>28} | {'N':>6} | {'WER':>8} | {'CER':>8}")
    print(f"  {'-'*57}")
    for r in summary_rows:
        label = f"{r['group_type']}:{r['group_value']}"
        print(f"  {label:>28} | {r['n_examples']:>6} | "
              f"{r['overall_wer']:>8.4f} | {r['overall_cer']:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("Evaluate Whisper (org NSS fine-tune) on Augmented NSS")
    print("=" * 65)
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Augmented dir : {args.augmented_dir}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Splits        : {EVAL_SPLITS}")
    print("=" * 65)

    model, processor = load_model_and_processor(args.checkpoint)
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    all_results = {}
    all_rows    = []

    for split in EVAL_SPLITS:
        rows, wer, cer = evaluate_split(
            split, args.augmented_dir,
            model, processor,
            wer_metric, cer_metric,
        )
        all_results[split] = (wer, cer)
        all_rows.extend(rows)

        save_csv(rows, os.path.join(args.output_dir, f"eval_{split}.csv"))
        save_summary(rows, split, args.output_dir)

    # ── Final results table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    # CDLI baselines (clean audio)
    cdli_clean = {"validation": 0.349, "test": 0.509}

    print(f"  {'Split':>12} | {'WER (aug)':>10} | {'CER (aug)':>10} | "
          f"{'CDLI clean':>12} | {'Diff':>8}")
    print(f"  {'-'*65}")
    for split, (wer, cer) in all_results.items():
        diff = wer - cdli_clean[split]
        sign = "+" if diff > 0 else ""
        print(f"  {split:>12} | {wer:>10.4f} | {cer:>10.4f} | "
              f"{cdli_clean[split]:>12.3f} | {sign}{diff:.4f}")

    print(f"\n  Note: CDLI baseline is on CLEAN audio.")
    print(f"  Aug WER is on clean+musan — higher WER on noisy is expected.")
    print(f"  For fair CDLI comparison use clean rows only:")
    print(f"    df[df.noise_type=='clean']['wer'].mean()")

    print(f"\n  Output files:")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        size  = os.path.getsize(fpath) // 1024
        print(f"    {fname}  ({size} KB)")


if __name__ == "__main__":
    main()