"""
NSS Detailed Evaluation Script — MMS
======================================
Evaluates best MMS NSS model on clean original NSS test and validation splits.
Saves per-example results to CSV including:
  - speaker_id, snr_db, snr_label, noise_type
  - original transcription, predicted transcription
  - per-example WER and CER
  - recording_environment, recording_device

Key differences from Whisper eval:
  - Uses Wav2Vec2ForCTC not WhisperForConditionalGeneration
  - CTC decoding: argmax → collapse repeats → remove blanks (no beam search)
  - Input: raw waveform (input_values) not mel spectrogram (input_features)
  - No forced_decoder_ids, no language token, no generate() — uses forward pass
  - Load local checkpoint WITHOUT target_lang (adapter baked into saved weights)

Usage:
  python3 eval_mms_nss.py
  python3 eval_mms_nss.py --checkpoint /lustre/scratch/cbr156l-my_gpu_project/train_mms_ga_nss_augmented_v2/best_model
"""

import io
import os
import re
import csv
import time
import argparse
import torch
import numpy as np
import soundfile as sf
import evaluate as hf_evaluate
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, AutoProcessor
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "/lustre/scratch/cbr156l-my_gpu_project/train_mms_ga_nss_augmented_v2/best_model"
BASE_MODEL         = "facebook/mms-1b-all"
LANG               = "aka"   # Akan — proxy for Ga
NSS_DATASET_ID     = "cdli/ghanian_ga_nonstandard_speech_v1.0"
AUGMENTED_DIR      = "/lustre/scratch/cbr156l-my_gpu_project/augmented_ga_nss_musan"
OUTPUT_DIR         = "eval_results_mms"
SR                 = 16000

# Ga-specific chars added during training
GA_SPECIFIC_CHARS  = ["ε", "ɔ", "ŋ", "Ɔ", "ɔ́", "ɔ̀", "ɛ́", "ɛ̀"]
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--augmented_dir", type=str, default=AUGMENTED_DIR)
    return p.parse_args()


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("<unk>", "").replace("unk", "")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def load_audio_wav(wav_path: str) -> np.ndarray:
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)


def transcribe(model, processor, array: np.ndarray):
    """
    MMS CTC inference — completely different from Whisper:

    Whisper: processor → input_features (mel spec) → model.generate() → tokens
    MMS:     processor → input_values (raw waveform) → model.forward() → logits
                       → argmax → processor.decode() (collapses repeats, removes blanks)

    Returns: (prediction_text, inference_time_seconds)
    """
    # Raw waveform input — no mel spectrogram
    inputs = processor(
        array, sampling_rate=SR, return_tensors="pt"
    ).to("cuda")

    t_start = time.perf_counter()
    with torch.no_grad():
        # CTC forward pass — single pass, not autoregressive
        logits = model(**inputs).logits
    t_end = time.perf_counter()

    # CTC greedy decode: argmax over vocab → collapse repeated tokens → remove blank
    ids = torch.argmax(logits, dim=-1)[0]
    pred = processor.decode(ids)
    pred = pred.replace("<unk>", "").strip()
    return pred, round(t_end - t_start, 4)


def wer_single(pred: str, ref: str, wer_metric) -> float:
    if not ref.strip():
        return 0.0
    try:
        return wer_metric.compute(predictions=[pred], references=[ref])
    except:
        return 1.0


def cer_single(pred: str, ref: str, cer_metric) -> float:
    if not ref.strip():
        return 0.0
    try:
        return cer_metric.compute(predictions=[pred], references=[ref])
    except:
        return 1.0


# ── CSV columns ───────────────────────────────────────────────────────────────
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


def evaluate_augmented_nss(split, model, processor, wer_metric, cer_metric, augmented_dir):
    """
    Evaluate MMS on augmented NSS (metadata.csv + WAV files).
    Same structure as Whisper eval but uses MMS transcribe().
    """
    import csv as csvlib
    print(f"\n[{split.upper()}] Loading augmented NSS from {augmented_dir}...")

    csv_path = os.path.join(augmented_dir, "metadata.csv")
    split_rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            if row["split"] == split:
                split_rows.append(row)

    print(f"  {len(split_rows):,} examples  (clean + musan)")

    rows = []
    all_refs, all_preds = [], []

    for i, meta_row in enumerate(split_rows):
        if i % 200 == 0:
            print(f"  [{i:>6}/{len(split_rows)}] ...")
        try:
            wav_path = os.path.join(
                augmented_dir, "audio", meta_row["split"], meta_row["audio_filename"]
            )
            array = load_audio_wav(wav_path)

            # MMS CTC inference
            pred, inf_time = transcribe(model, processor, array)
            ref = meta_row.get("transcription", "")

            pred_norm = normalize(pred)
            ref_norm  = normalize(ref)

            wer = wer_single(pred_norm, ref_norm, wer_metric)
            cer = cer_single(pred_norm, ref_norm, cer_metric)

            all_refs.append(ref_norm)
            all_preds.append(pred_norm)

            snr_db    = float(meta_row.get("snr_db", 999))
            audio_len = float(meta_row.get("audio_length") or 0)
            rtf       = round(inf_time / audio_len, 4) if audio_len > 0 else ""

            rows.append({
                "split":                 split,
                "example_idx":           i,
                "speaker_id":            meta_row.get("speaker_id", ""),
                "recording_environment": meta_row.get("recording_environment", ""),
                "recording_device":      meta_row.get("recording_device", ""),
                "snr_db":                snr_db,
                "snr_label":             meta_row.get("snr_label", "clean"),
                "noise_type":            meta_row.get("noise_type", "clean"),
                "audio_length":          meta_row.get("audio_length", ""),
                "transcript_length":     meta_row.get("transcript_length", ""),
                "reference":             ref,
                "prediction":            pred,
                "reference_normalized":  ref_norm,
                "prediction_normalized": pred_norm,
                "wer":                   round(wer, 4),
                "cer":                   round(cer, 4),
                "correct":               1 if wer == 0.0 else 0,
                "inference_time_s":      inf_time,
                "rtf":                   rtf,
            })
        except Exception as e:
            print(f"  Skip {i}: {e}")

    overall_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    overall_cer = cer_metric.compute(predictions=all_preds, references=all_refs)
    print(f"\n  [{split.upper()}] Overall WER: {overall_wer:.4f}  CER: {overall_cer:.4f}")

    return rows, overall_wer, overall_cer


def save_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLS})
    print(f"  Saved: {path}  ({len(rows):,} rows)")


def save_summary(rows: list, split_name: str, output_dir: str):
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    def make_group(label_type, label_value, d):
        if not d["refs"]:
            return None
        return {
            "split":       split_name,
            "group_type":  label_type,
            "group_value": label_value,
            "n_examples":  len(d["refs"]),
            "overall_wer": round(wer_metric.compute(
                               predictions=d["preds"], references=d["refs"]), 4),
            "overall_cer": round(cer_metric.compute(
                               predictions=d["preds"], references=d["refs"]), 4),
            "avg_wer":     round(np.mean(d["wers"]), 4),
            "avg_cer":     round(np.mean(d["cers"]), 4),
        }

    def empty():
        return {"refs": [], "preds": [], "wers": [], "cers": []}

    spk_data = defaultdict(empty)
    snr_data = defaultdict(empty)
    env_data = defaultdict(empty)

    for row in rows:
        spk = str(row.get("speaker_id", "unknown"))
        snr = str(row.get("snr_label", "clean"))
        env = str(row.get("recording_environment", "unknown"))
        for d in [spk_data[spk], snr_data[snr], env_data[env]]:
            d["refs"].append(row["reference_normalized"])
            d["preds"].append(row["prediction_normalized"])
            d["wers"].append(float(row["wer"]))
            d["cers"].append(float(row["cer"]))

    summary_rows = []
    for spk in sorted(spk_data):
        r = make_group("speaker", spk, spk_data[spk])
        if r: summary_rows.append(r)

    for snr in ["clean", "-10dB", "-5dB", "+0dB", "+5dB", "+10dB", "+20dB"]:
        if snr in snr_data:
            r = make_group("snr", snr, snr_data[snr])
            if r: summary_rows.append(r)

    for env in sorted(env_data):
        r = make_group("environment", env, env_data[env])
        if r: summary_rows.append(r)

    summary_cols = ["split", "group_type", "group_value", "n_examples",
                    "overall_wer", "overall_cer", "avg_wer", "avg_cer"]

    path = os.path.join(output_dir, f"summary_{split_name}.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved: {path}  ({len(summary_rows)} groups)")

    print(f"\n  [{split_name.upper()}] Summary:")
    print(f"  {'Group':>25} | {'N':>6} | {'WER':>8} | {'CER':>8}")
    print(f"  {'-'*55}")
    for r in summary_rows:
        print(f"  {r['group_type']+':'+r['group_value']:>25} | "
              f"{r['n_examples']:>6} | {r['overall_wer']:>8.4f} | {r['overall_cer']:>8.4f}")
    return summary_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(f"MMS NSS Detailed Evaluation")
    print("=" * 65)
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Base model    : {BASE_MODEL}")
    print(f"  Language      : {LANG} (Akan proxy for Ga)")
    print(f"  Augmented dir : {args.augmented_dir}")
    print(f"  Output        : {args.output_dir}/")
    print("=" * 65)

    # ── Load processor from BASE_MODEL with target_lang ───────────────────────
    # Must load processor from HuggingFace (not local checkpoint) to get
    # correct tokenizer config, then add Ga-specific characters
    print(f"\nLoading processor from: {BASE_MODEL}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, target_lang=LANG)

    # Add Ga-specific characters (same as training)
    current_vocab = processor.tokenizer.get_vocab()
    missing = [c for c in GA_SPECIFIC_CHARS if c not in current_vocab]
    if missing:
        print(f"  Adding {len(missing)} Ga chars: {missing}")
        processor.tokenizer.add_tokens(missing)
    print(f"  Vocab size: {len(processor.tokenizer)}")

    # ── Load model from LOCAL checkpoint ──────────────────────────────────────
    # Do NOT pass target_lang here — adapter weights are already baked into
    # the saved checkpoint. Passing target_lang would trigger load_adapter()
    # which looks for adapter.aka.bin file that doesn't exist locally.
    print(f"\nLoading model from: {args.checkpoint}")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.checkpoint,
        ignore_mismatched_sizes=True,
    ).cuda()
    model.eval()
    print(f"  Model loaded ✓")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    all_rows    = []
    all_results = {}

    for split in ["validation", "test"]:
        rows, wer, cer = evaluate_augmented_nss(
            split, model, processor, wer_metric, cer_metric, args.augmented_dir
        )
        all_rows.extend(rows)
        all_results[split] = (wer, cer)
        save_csv(rows, os.path.join(args.output_dir, f"eval_{split}.csv"))

    for split, rows in [
        ("validation", [r for r in all_rows if r["split"] == "validation"]),
        ("test",       [r for r in all_rows if r["split"] == "test"]),
    ]:
        if rows:
            save_summary(rows, split, args.output_dir)

    # ── Final results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    cdli = {"validation": (0.349, 0.159), "test": (0.509, 0.293)}
    print(f"  {'Split':>12} | {'WER':>8} | {'CER':>8} | {'CDLI WER':>10} | {'Diff':>8}")
    print(f"  {'-'*60}")
    for split, (wer, cer) in all_results.items():
        cdli_wer = cdli[split][0]
        diff     = wer - cdli_wer
        sign     = "+" if diff > 0 else ""
        print(f"  {split:>12} | {wer:>8.4f} | {cer:>8.4f} | "
              f"{cdli_wer:>10.3f} | {sign}{diff:.4f}")

    print(f"\n  Output files:")
    for fname in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, fname)
        size  = os.path.getsize(fpath)
        print(f"    {fname}  ({size//1024} KB)")


if __name__ == "__main__":
    main()
