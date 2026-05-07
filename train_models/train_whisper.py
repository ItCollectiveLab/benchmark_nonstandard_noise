"""
Whisper Large-v3 Fine-Tuning for Ghanaian Ga
=============================================
Two-stage pipeline:

  STAGE 1 — Standard Speech (SS):
    python3 train_whisper.py \
        --mode standard \
        --output_dir train_whisp_ga_large_v3_ss \
        --max_steps 1000

  STAGE 2 — Non-Standard Speech (NSS) with augmentation:
    python3 train_whisper.py \
        --mode nonstandard \
        --ss_model_dir train_whisp_ga_large_v3_ss/checkpoint-600 \
        --nss_data_dir augmented_ga_nss_musan \
        --output_dir train_whisp_ga_large_v3_nss_augmented \
        --max_steps 1000

RAM fix: augmented NSS WAVs are loaded on-the-fly per batch,
         NOT all preloaded into memory. Only 4 examples in RAM at a time.
"""

import os
import csv
import io
import argparse
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset, Audio, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# ─────────────────────────────────────────────
# 0. CONSTANTS
# ─────────────────────────────────────────────

BASE_MODEL = "openai/whisper-large-v3"
LANGUAGE   = "yo"
TASK       = "transcribe"

NUM_MEL_BINS        = 128
MASK_TIME_PROB      = 0.05
MASK_TIME_LENGTH    = 10
MASK_TIME_MIN_MASKS = 2
MASK_FEAT_PROB      = 0.05
MASK_FEAT_LENGTH    = 10
MASK_FEAT_MIN_MASKS = 2

BEGIN_SUPPRESS  = [220, 50256]
SUPPRESS_TOKENS = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
    90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350,
    1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
    6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
    13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075,
    21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
    36865, 42863, 47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363,
]

DS_STANDARD    = "cdli/ghanian_ga_standard_speech_v1.0"
DS_NONSTANDARD = "cdli/ghanian_ga_nonstandard_speech_v1.0"

# ─────────────────────────────────────────────
# 1. ARGS
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standard", "nonstandard"], required=True)
    p.add_argument("--ss_model_dir", type=str, default=None)
    p.add_argument("--nss_data_dir", type=str, default=None,
                   help="Path to augmented NSS dir with metadata.csv + audio/")
    p.add_argument("--output_dir",   type=str, default=None)
    p.add_argument("--max_steps",    type=int, default=1000)
    p.add_argument("--resume_from",  type=str, default=None)
    p.add_argument("--push_to_hub",  action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)
    return p.parse_args()

# ─────────────────────────────────────────────
# 2. PROCESSOR
# ─────────────────────────────────────────────

def load_processor(model_dir=None):
    source = model_dir if model_dir else BASE_MODEL
    print(f"Loading processor from: {source}")
    processor = WhisperProcessor.from_pretrained(
        source, language=LANGUAGE, task=TASK
    )
    actual_bins = processor.feature_extractor.feature_size
    assert actual_bins == NUM_MEL_BINS, (
        f"Expected {NUM_MEL_BINS} mel bins, got {actual_bins}"
    )
    print(f"  Mel bins : {actual_bins} ✓")
    print(f"  Language : {LANGUAGE}  (Yoruba — closest Kwa language to Ga)")
    return processor

# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────

def load_model(mode, ss_model_dir=None):
    is_standard = (mode == "standard")

    if is_standard:
        print(f"\nSS: Loading base model: {BASE_MODEL}")
        model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16
        )
    else:
        if ss_model_dir is None:
            raise ValueError("--ss_model_dir is required for nonstandard mode.")
        print(f"\nNSS: Loading SS checkpoint: {ss_model_dir}")
        model = WhisperForConditionalGeneration.from_pretrained(
            ss_model_dir, torch_dtype=torch.bfloat16
        )

    model.config.apply_spec_augment     = is_standard
    model.config.mask_time_prob         = MASK_TIME_PROB      if is_standard else 0.0
    model.config.mask_time_length       = MASK_TIME_LENGTH
    model.config.mask_time_min_masks    = MASK_TIME_MIN_MASKS
    model.config.mask_feature_prob      = MASK_FEAT_PROB      if is_standard else 0.0
    model.config.mask_feature_length    = MASK_FEAT_LENGTH
    model.config.mask_feature_min_masks = MASK_FEAT_MIN_MASKS
    model.config.use_cache              = False
    model.config.forced_decoder_ids     = None
    model.config.suppress_tokens        = SUPPRESS_TOKENS
    model.config.begin_suppress_tokens  = BEGIN_SUPPRESS
    model.generation_config.language    = LANGUAGE
    model.generation_config.task        = TASK
    model.generation_config.return_timestamps = False

   # if not is_standard:
      #  print("NSS: Freezing encoder — only decoder will be fine-tuned")
       # for param in model.model.encoder.parameters():
         #   param.requires_grad = False

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SpecAugment      : {model.config.apply_spec_augment}")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}  ({'all' if is_standard else 'decoder only'})")
    print(f"  Frozen params    : {total - trainable:,}  ({'none' if is_standard else 'encoder'})")
    return model

# ─────────────────────────────────────────────
# 4. LAZY DATASET — key fix for RAM overflow
# ─────────────────────────────────────────────

class LazyAugmentedNSSDataset(TorchDataset):
    """
    Loads WAV files on-the-fly during training.
    Only 1 example is in RAM per __getitem__ call.
    Memory usage: ~constant (batch_size × example_size) instead of
                  77,056 × example_size = ~118 GB.

    __len__  → returns total number of examples (77,056 for train)
    __getitem__ → reads ONE wav file from disk, extracts features, returns dict
    """

    def __init__(self, rows: list, nss_data_dir: str,
                 split_name: str, processor: Any):
        self.rows         = rows
        self.nss_data_dir = nss_data_dir
        self.split_name   = split_name
        self.processor    = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        # Read ONE wav file from disk — not cached, not pre-loaded
        wav_path = os.path.join(
            self.nss_data_dir, "audio", self.split_name, row["audio_filename"]
        )
        array, sr = sf.read(wav_path)
        if array.ndim > 1:
            array = array.mean(axis=1)       # stereo → mono
        if sr != 16000:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        array = array.astype(np.float32)

        # Extract mel spectrogram features
        feats = self.processor.feature_extractor(
            array, sampling_rate=16000
        ).input_features[0]

        # Tokenize transcription
        labels = self.processor.tokenizer(
            row["transcription"]
        ).input_ids

        return {"input_features": feats, "labels": labels}

# ─────────────────────────────────────────────
# 5. DATA COLLATOR
# ─────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# ─────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────

def build_compute_metrics(processor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4),
            "cer": round(cer_metric.compute(predictions=pred_str, references=label_str), 4),
        }
    return compute_metrics

# ─────────────────────────────────────────────
# 7. PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def audio_bytes_to_array(audio_bytes):
    array, sr = sf.read(io.BytesIO(audio_bytes))
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
    return array.astype(np.float32)


def build_preprocess_fn(processor, transcription_col):
    def preprocess(batch):
        array = audio_bytes_to_array(batch["audio"]["bytes"])
        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            batch[transcription_col]
        ).input_ids
        return batch
    return preprocess

# ─────────────────────────────────────────────
# 8. DATASET LOADERS
# ─────────────────────────────────────────────

def load_hf_dataset(mode, processor):
    """Load original HuggingFace dataset (SS or NSS without augmentation)."""
    ds_id = DS_STANDARD if mode == "standard" else DS_NONSTANDARD
    print(f"\nLoading HF dataset: {ds_id}")
    ds = load_dataset(ds_id)
    ds = ds.cast_column("audio", Audio(decode=False))

    for split in ds:
        print(f"  [{split}] {len(ds[split]):,} examples")

    tx_col = next(
        (c for c in ["transcription", "sentence", "text", "transcript"]
         if c in ds[list(ds.keys())[0]].features),
        None
    )
    print(f"  Transcription column: '{tx_col}'")

    cols_to_remove = [
        c for c in ds[list(ds.keys())[0]].features
        if c not in ["input_features", "labels"]
    ]

    print("\nExtracting features...")
    ds = ds.map(
        build_preprocess_fn(processor, tx_col),
        remove_columns=cols_to_remove,
        desc="Preprocessing",
    )

    train_ds = ds["train"]
    eval_ds  = ds["validation"] if "validation" in ds else ds["dev"]
    test_ds  = ds.get("test")

    print(f"  Train : {len(train_ds):,}")
    print(f"  Eval  : {len(eval_ds):,}")
    if test_ds:
        print(f"  Test  : {len(test_ds):,}")

    return train_ds, eval_ds, test_ds


def load_augmented_nss(nss_data_dir, processor):
    """
    Load augmented NSS using LAZY loading — WAVs read on-the-fly.
    RAM usage: constant ~few MB regardless of dataset size.

    Only the CSV metadata is loaded into memory (tiny — ~30MB for 85k rows).
    WAV files are read from disk one at a time during training.
    """
    print(f"\nLoading augmented NSS (lazy): {nss_data_dir}")
    csv_path = os.path.join(nss_data_dir, "metadata.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metadata.csv not found in {nss_data_dir}")

    # Only the CSV rows go into memory — not the audio
    split_rows = {"train": [], "dev": [], "test": [], "validation": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split in split_rows:
                split_rows[split].append(row)

    if not split_rows["dev"] and split_rows["validation"]:
        split_rows["dev"] = split_rows["validation"]

    for split, rows in split_rows.items():
        if rows:
            clean = sum(1 for r in rows if r.get("noise_type") == "clean")
            noisy = sum(1 for r in rows if r.get("noise_type") == "musan")
            print(f"  [{split}] {len(rows):,} total  ({clean:,} clean + {noisy:,} noisy)")

    # Create lazy datasets — no WAV loading happens here
    print("\n  Creating lazy datasets (no preprocessing — WAVs load on-the-fly)...")

    train_ds = LazyAugmentedNSSDataset(
        split_rows["train"], nss_data_dir, "train", processor
    )

    # Use validation if it has data, else fall back to dev
    if split_rows["validation"]:
        eval_rows      = split_rows["validation"]
        eval_split_dir = "validation"
    else:
        eval_rows      = split_rows["dev"]
        eval_split_dir = "dev"
    eval_ds = LazyAugmentedNSSDataset(
        eval_rows, nss_data_dir, eval_split_dir, processor
    )

    test_ds = None
    if split_rows["test"]:
        test_ds = LazyAugmentedNSSDataset(
            split_rows["test"], nss_data_dir, "test", processor
        )

    print(f"\n  Train : {len(train_ds):,} examples  (WAVs load on-the-fly)")
    print(f"  Eval  : {len(eval_ds):,} examples  (WAVs load on-the-fly)")
    if test_ds:
        print(f"  Test  : {len(test_ds):,} examples  (WAVs load on-the-fly)")
    print(f"  RAM used by metadata only: ~{sum(len(r) for r in split_rows['train'])//1024//1024 + 1} MB")

    return train_ds, eval_ds, test_ds

# ─────────────────────────────────────────────
# 9. TRAINING ARGUMENTS
# ─────────────────────────────────────────────

def get_training_args(output_dir, max_steps, push_to_hub, hub_model_id):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=100,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=False,
        bf16=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=50,
        weight_decay=0.0,
        lr_scheduler_type="polynomial",
        lr_scheduler_kwargs={"lr_end": 1e-8, "power": 4},
        logging_steps=50,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        remove_unused_columns=False,
        # Use multiple workers to load WAVs in parallel while GPU trains
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    if args.mode == "nonstandard" and args.ss_model_dir is None:
        raise ValueError(
            "ERROR: --ss_model_dir is required for nonstandard mode."
        )

    if args.output_dir is None:
        suffix = "ss" if args.mode == "standard" else "nss"
        if args.nss_data_dir:
            suffix += "_augmented"
        args.output_dir = f"train_whisp_ga_large_v3_{suffix}"

    print("=" * 65)
    print(f"MODE         : {args.mode}")
    print(f"OUTPUT       : {args.output_dir}")
    print(f"MAX STEPS    : {args.max_steps}")
    if args.mode == "nonstandard":
        print(f"SS MODEL     : {args.ss_model_dir}")
        print(f"NSS DATA     : {args.nss_data_dir or 'HuggingFace (no augmentation)'}")
    print("=" * 65)

    processor = load_processor(
        model_dir=args.ss_model_dir if args.mode == "nonstandard" else None
    )
    model = load_model(mode=args.mode, ss_model_dir=args.ss_model_dir)

    if args.mode == "standard":
        train_ds, eval_ds, test_ds = load_hf_dataset("standard", processor)
    elif args.nss_data_dir:
        train_ds, eval_ds, test_ds = load_augmented_nss(args.nss_data_dir, processor)
    else:
        train_ds, eval_ds, test_ds = load_hf_dataset("nonstandard", processor)

    print(f"\nFinal dataset sizes:")
    print(f"  Train : {len(train_ds):,}")
    print(f"  Eval  : {len(eval_ds):,}")
    if test_ds:
        print(f"  Test  : {len(test_ds):,}")

    data_collator   = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = build_compute_metrics(processor)
    training_args   = get_training_args(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print(f"\nStarting {args.mode} training — {args.max_steps} steps...")
    if args.mode == "standard":
        print("  Full model (encoder + decoder) | SpecAugment ON")
    else:
        #print("  Encoder FROZEN | decoder only | SpecAugment OFF")
        if args.nss_data_dir:
            print(f"  Lazy loading: WAVs read from disk on-the-fly")
            print(f"  RAM for audio: ~0 MB upfront (loaded per batch)")

    trainer.train(resume_from_checkpoint=args.resume_from)

    best_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_path)
    processor.save_pretrained(best_path)
    print(f"\nBest model saved: {best_path}")

    print("\n" + "=" * 65)
    print("FINAL EVALUATION")
    print("=" * 65)

    eval_sets = {"validation/dev": eval_ds}
    if test_ds:
        eval_sets["test"] = test_ds

    for split_name, split_ds in eval_sets.items():
        r = trainer.evaluate(eval_dataset=split_ds)
        print(f"[{split_name.upper():>15}]  "
              f"WER={r.get('eval_wer', float('nan')):.4f}  "
              f"CER={r.get('eval_cer', float('nan')):.4f}  "
              f"Loss={r.get('eval_loss', float('nan')):.4f}")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()