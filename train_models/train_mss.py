"""
MMS (facebook/mms-1b-all) Fine-Tuning for Ghanaian Ga (NSS)
=============================================================
Two-stage pipeline:

  STAGE 1 — Standard Speech (SS):
    python3 train_mms.py \
        --lang aka \
        --mode standard \
        --output_dir train_mms_ga_ss \
        --max_steps 1000

  STAGE 2 — Non-Standard Speech (NSS, no augmentation):
    python3 train_mms.py \
        --lang aka \
        --mode nonstandard \
        --ss_model_dir train_mms_ga_ss/best_model \
        --output_dir train_mms_ga_nss \
        --max_steps 2000

  STAGE 2b — Non-Standard Speech (augmented):
    python3 train_mms.py \
        --lang aka \
        --mode nonstandard \
        --ss_model_dir train_mms_ga_ss/best_model \
        --nss_data_dir /path/to/augmented_ga_nss_musan \
        --output_dir train_mms_ga_nss_augmented \
        --max_steps 2000

Architecture notes:
  - Ga (gaa) has no MMS ASR support → use aka (Akan), same Kwa family
  - MMS uses CTC loss, NOT seq2seq like Whisper
  - Input: raw waveform (input_values), NOT mel spectrogram
  - SS:  FULL model fine-tuning (all 1B params) — matches Whisper SS strategy
  - NSS: adapter + lm_head only — preserve SS Ga acoustic representations
"""

import os
import csv
import argparse
import torch
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
import evaluate

# ─────────────────────────────────────────────
# 0. CONSTANTS
# ─────────────────────────────────────────────

BASE_MODEL     = "facebook/mms-1b-all"
DS_STANDARD    = "cdli/ghanian_ga_standard_speech_v1.0"
DS_NONSTANDARD = "cdli/ghanian_ga_nonstandard_speech_v1.0"

# Ga-specific characters not in Akan (aka) tokenizer vocab
GA_SPECIFIC_CHARS = ["ε", "ɔ", "ŋ", "Ɔ", "ɔ́", "ɔ̀", "ɛ́", "ɛ̀"]

# ─────────────────────────────────────────────
# 1. ARGS
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lang", type=str, required=True,
        help=(
            "ISO 639-3 language code.\n"
            "Ga (gaa) has no MMS ASR support.\n"
            "Use 'aka' (Akan) — same Kwa family, closest available.\n"
            "Other options: ewe, lug (Luganda), swh (Swahili), eng"
        )
    )
    p.add_argument(
        "--mode", choices=["standard", "nonstandard"], default="standard",
        help=(
            "standard    → FULL model fine-tune from MMS base (aka adapter)\n"
            "nonstandard → adapter + lm_head only, from --ss_model_dir"
        )
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="HuggingFace dataset ID. Defaults to CDLI SS or NSS based on mode."
    )
    p.add_argument(
        "--ss_model_dir", type=str, default=None,
        help="Path to SS best_model directory. REQUIRED for --mode nonstandard."
    )
    p.add_argument(
        "--nss_data_dir", type=str, default=None,
        help=(
            "Path to augmented NSS directory (produced by aug_data.py).\n"
            "If not set, uses original HF NSS dataset.\n"
            "Expected structure:\n"
            "  nss_data_dir/\n"
            "    metadata.csv\n"
            "    audio/\n"
            "      train/\n"
            "      dev/\n"
            "      test/"
        )
    )
    p.add_argument("--output_dir",   type=str, default=None)
    p.add_argument("--max_steps",    type=int, default=1000)
    p.add_argument("--resume_from",  type=str, default=None)
    p.add_argument("--push_to_hub",  action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)
    return p.parse_args()

# ─────────────────────────────────────────────
# 2. PROCESSOR + MODEL
# ───────────────────────────────────────
def load_processor_and_model(lang: str, mode: str, ss_model_dir: str = None):

    print(f"\nLoading MMS processor from: {BASE_MODEL}")
    # ── Official MMS loading — target_lang handles adapter + lm_head resize
    processor = AutoProcessor.from_pretrained(BASE_MODEL, target_lang=lang)
    print(f"  Tokenizer language : {lang}")

    if mode == "standard":
        if ss_model_dir:
            print(f"SS: Continuing from checkpoint: {ss_model_dir}")
            model = Wav2Vec2ForCTC.from_pretrained(
                ss_model_dir,
                ignore_mismatched_sizes=True,
                dtype=torch.float32,          # note: use dtype not torch_dtype
            )

        else:
            print(f"SS: Loading base model: {BASE_MODEL}")
            model = Wav2Vec2ForCTC.from_pretrained(
                BASE_MODEL,
                target_lang=lang,
                ignore_mismatched_sizes=True,
                dtype=torch.float32,
            )
        model.config.apply_spec_augment      = True
        model.config.mask_time_prob          = 0.05
        model.config.mask_time_length        = 10
        model.config.mask_time_min_masks     = 2
        model.config.mask_feature_prob       = 0.05
        model.config.mask_feature_length     = 10
        model.config.mask_feature_min_masks  = 2 
    else:
        if ss_model_dir is None:
            raise ValueError("--ss_model_dir is required for nonstandard mode.")
        print(f"NSS: Loading SS checkpoint: {ss_model_dir}")
        model = Wav2Vec2ForCTC.from_pretrained(
            ss_model_dir,
            ignore_mismatched_sizes=True,
            dtype=torch.float32,
        )
       # model.config.apply_spec_augment      = True
       # model.config.mask_time_prob          = 0.03
       # model.config.mask_time_length        = 10
       # model.config.mask_time_min_masks     = 1
       # model.config.mask_feature_prob       = 0.03
       # model.config.mask_feature_length     = 10
       # model.config.mask_feature_min_masks  = 1

    print(f"  Loaded adapter: {lang}")

    # ── Add missing Ga characters to vocab ───────────────────────────
    current_vocab = processor.tokenizer.get_vocab()
    missing_chars = [c for c in GA_SPECIFIC_CHARS if c not in current_vocab]

    if missing_chars:
        print(f"\n  Adding {len(missing_chars)} missing Ga chars: {missing_chars}")
        processor.tokenizer.add_tokens(missing_chars)
        new_vocab_size = len(processor.tokenizer)

        old_lm_head = model.lm_head
        model.lm_head = torch.nn.Linear(
            old_lm_head.in_features, new_vocab_size, bias=True
        )
        with torch.no_grad():
            model.lm_head.weight[:old_lm_head.out_features] = old_lm_head.weight
            model.lm_head.bias[:old_lm_head.out_features]   = old_lm_head.bias
            torch.nn.init.xavier_uniform_(
                model.lm_head.weight[old_lm_head.out_features:]
            )
        model.config.vocab_size = new_vocab_size
        print(f"  lm_head resized: {old_lm_head.out_features} → {new_vocab_size}")
    else:
        print(f"  All Ga characters already in vocab ✓")

    # ── Freeze strategy ───────────────────────────────────────────────
   # for param in model.parameters():
    #    param.requires_grad = True
    if mode == "standard":
        for param in model.parameters():
            param.requires_grad = True
        print("\n  SS: FULL fine-tuning — all parameters unfrozen")
    else:
        model.freeze_base_model()
        for name, param in model.named_parameters():
            if any(k in name for k in [
                "adapter",
                "lm_head",
                "wav2vec2.encoder.layers.23",
                "wav2vec2.encoder.layers.22",
                "wav2vec2.encoder.layers.21",
                "wav2vec2.encoder.layers.20",
            ]):
                param.requires_grad = True
        print("\n  NSS:4 top encoder +  adapter + lm_head only (base encoder frozen)")
   # print("\n  SS: FULL fine-tuning — all parameters unfrozen")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen params    : {total-trainable:,}  ({100*(total-trainable)/total:.1f}%)")

    return processor, model
# ─────────────────────────────────────────────
# 3. LAZY DATASET
# ─────────────────────────────────────────────

class LazyAugmentedNSSDataset(TorchDataset):
    """
    Loads WAV files on-the-fly during training.
    Only 1 example in RAM per __getitem__ call — same RAM strategy as Whisper code.

    Key difference from Whisper LazyAugmentedNSSDataset:
      - Returns input_values (raw waveform float32 array)
        instead of input_features (128-bin mel spectrogram)
      - Labels are CTC character-level token IDs
        instead of Whisper subword token IDs
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

        wav_path = os.path.join(
            self.nss_data_dir, "audio", self.split_name, row["audio_filename"]
        )
        array, sr = sf.read(wav_path)
        if array.ndim > 1:
            array = array.mean(axis=1)           # stereo → mono
        if sr != 16000:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        array = array.astype(np.float32)

        # Raw waveform — no mel spectrogram extraction
        inputs = self.processor(array, sampling_rate=16000)
        item   = {"input_values": inputs.input_values[0]}

        # CTC character-level label encoding
        #with self.processor.as_target_processor():
         #   item["labels"] = self.processor(row["transcription"]).input_ids
        item["labels"] = self.processor.tokenizer(row["transcription"]).input_ids
        return item

# ─────────────────────────────────────────────
# 4. DATA COLLATOR
# ─────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    """
    MMS CTC data collator.

    Differences from Whisper DataCollatorSpeechSeq2SeqWithPadding:

    1. Pads input_values (variable-length raw audio waveforms)
       Whisper pads input_features (fixed 30s mel spectrograms — no padding needed)

    2. Labels padded with -100 for CTCLoss ignore_index
       Whisper also uses -100 but for cross-entropy on decoder outputs

    3. No BOS token stripping
       CTC has no autoregressive decoder — labels are just character sequences
       Whisper must strip the leading BOS because the decoder uses it as a prompt

    4. Uses processor.as_target_processor() context for label padding
       Required for MMS because processor has dual mode (audio vs text)
    """
    processor: Any
    padding:   bool = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # Pad raw waveforms — lengths vary per example
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad CTC labels with -100 (CTCLoss ignores these positions)
        label_features = [{"input_ids": f["labels"]} for f in features]
        #with self.processor.as_target_processor():
         #   labels_batch = self.processor.pad(
          #      label_features,
           #     padding=self.padding,
            #    return_tensors="pt",
            #)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features,padding=self.padding,return_tensors="pt",)

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

# ─────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────

def build_compute_metrics(processor):
    """
    WER and CER — same metrics as Whisper code.

    CTC decoding difference:
      - pred_ids = argmax(logits) then processor.batch_decode() collapses
        repeated tokens and removes blank tokens automatically
      - Whisper uses beam search via generate() — richer but slower
      - group_tokens=False on label decode prevents collapsing real
        repeated characters in Ga reference transcriptions
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids    = np.argmax(pred_logits, axis=-1)

        # CTC greedy decode: collapse repeats, remove blanks
        pred_str = processor.batch_decode(pred_ids)

        # Decode reference labels — group_tokens=False keeps real repetitions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        #label_str = processor.batch_decode(label_ids, group_tokens=False)
        label_str = processor.tokenizer.batch_decode(label_ids, group_tokens=False)

        return {
            "wer": round(wer_metric.compute(
                predictions=pred_str, references=label_str), 4),
            "cer": round(cer_metric.compute(
                predictions=pred_str, references=label_str), 4),
        }
    return compute_metrics

# ─────────────────────────────────────────────
# 6. AUDIO HELPERS
# ─────────────────────────────────────────────

def audio_bytes_to_array(audio_bytes: bytes) -> np.ndarray:
    """Decode raw audio bytes → float32 numpy array at 16kHz."""
    import io
    array, sr = sf.read(io.BytesIO(audio_bytes))
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
    return array.astype(np.float32)


def wav_path_to_array(wav_path: str) -> np.ndarray:
    """Read WAV file from disk → float32 numpy array at 16kHz."""
    array, sr = sf.read(wav_path)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
    return array.astype(np.float32)

# ─────────────────────────────────────────────
# 7. DATASET LOADERS
# ─────────────────────────────────────────────

def load_hf_dataset(dataset_id: str, processor):
    """
    Load and preprocess a HuggingFace dataset for MMS.

    Key difference from Whisper load_hf_dataset:
      - processor() returns input_values (raw waveform)
        Whisper uses processor.feature_extractor() for mel spectrogram
      - Labels encoded with processor.as_target_processor() context
        Whisper uses processor.tokenizer() directly
    """
    print(f"\nLoading HF dataset: {dataset_id}")
    ds = load_dataset(dataset_id)
    ds = ds.cast_column("audio", Audio(decode=False))

    for split in ds:
        print(f"  [{split}] {len(ds[split]):,} examples")

    tx_col = next(
        (c for c in ["transcription", "sentence", "text", "transcript"]
         if c in ds[list(ds.keys())[0]].features),
        None
    )
    print(f"  Transcription column: '{tx_col}'")

    def preprocess(batch):
        array = audio_bytes_to_array(batch["audio"]["bytes"])

        # Raw waveform input — no mel spectrogram
        inputs = processor(array, sampling_rate=16000)
        batch["input_values"] = inputs.input_values[0]

        # CTC label encoding requires as_target_processor context
        #with processor.as_target_processor():
         #   batch["labels"] = processor(batch[tx_col]).input_ids

        batch["labels"] = processor.tokenizer(batch[tx_col]).input_ids
        return batch

    cols_to_remove = [
        c for c in ds[list(ds.keys())[0]].features
        if c not in ["input_values", "labels"]
    ]

    print("\nExtracting features...")
    ds = ds.map(
        preprocess,
        remove_columns=cols_to_remove,
        desc="Preprocessing"
    )

    train_ds = ds["train"]
    eval_ds  = ds.get("validation") or ds.get("dev")
    test_ds  = ds.get("test")

    print(f"  Train : {len(train_ds):,}")
    print(f"  Eval  : {len(eval_ds):,}")
    if test_ds:
        print(f"  Test  : {len(test_ds):,}")

    return train_ds, eval_ds, test_ds


def load_augmented_nss(nss_data_dir: str, processor):
    """
    Load augmented NSS dataset with lazy WAV loading.
    Only metadata CSV goes into memory — WAVs read from disk on-the-fly.
    Same lazy strategy as Whisper code to prevent RAM overflow.
    """
    print(f"\nLoading augmented NSS (lazy): {nss_data_dir}")
    csv_path = os.path.join(nss_data_dir, "metadata.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metadata.csv not found in {nss_data_dir}")

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

    train_ds = LazyAugmentedNSSDataset(
        split_rows["train"], nss_data_dir, "train", processor
    )

    eval_rows      = split_rows["validation"] if split_rows["validation"] else split_rows["dev"]
    eval_split_dir = "validation" if split_rows["validation"] else "dev"
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

    return train_ds, eval_ds, test_ds

# ─────────────────────────────────────────────
# 8. TRAINING ARGUMENTS
# ─────────────────────────────────────────────

def get_training_args(output_dir, max_steps, mode, push_to_hub, hub_model_id):
    """
    Key differences from Whisper Seq2SeqTrainingArguments:

    1. TrainingArguments not Seq2SeqTrainingArguments
       CTC has no autoregressive decoder → predict_with_generate not applicable

    2. fp16=True works fine for MMS
       Whisper requires bf16=True due to numerical stability in seq2seq decoding

    3. Higher LR for SS (full fine-tune) vs NSS (partial)
       SS trains 1B params from scratch → needs stronger gradient signal
       NSS trains only adapters → smaller LR prevents overshooting

    4. No generation_max_length
       CTC decoding length is determined by encoder output, not generation config
    """
    lr           =1e-4 #1e-4   # full fine-tune — same as Whisper SS
    warmup_steps =50 #50
    weight_decay =0.01  #0.0

    return TrainingArguments(
        output_dir=output_dir,

        # Steps
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

        # Precision — fp16 fine for MMS CTC
        fp16=True,
        bf16=False,

        # Batch / optimizer
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,   # effective batch = 32
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",

        # Logging / memory
        logging_steps=50,
        gradient_checkpointing=True,     # essential for 1B model on 16GB GPU
        report_to=["tensorboard"],
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate
    if args.mode == "nonstandard" and args.ss_model_dir is None:
        raise ValueError(
            "ERROR: --ss_model_dir is required for nonstandard mode.\n"
            "Train standard speech first:\n"
            "  python3 train_mms.py \\\n"
            "      --lang aka \\\n"
            "      --mode standard \\\n"
            "      --output_dir train_mms_ga_ss \\\n"
            "      --max_steps 1000\n"
            "Then pass: --ss_model_dir train_mms_ga_ss/best_model"
        )

    # Default output dir
    if args.output_dir is None:
        suffix = "ss" if args.mode == "standard" else "nss"
        if args.nss_data_dir:
            suffix += "_augmented"
        args.output_dir = f"train_mms_{args.lang}_{suffix}"

    # Default dataset
    if args.dataset is None:
        args.dataset = DS_STANDARD if args.mode == "standard" else DS_NONSTANDARD

    print("=" * 65)
    print(f"MODEL        : {BASE_MODEL}")
    print(f"LANGUAGE     : {args.lang}  (Akan — Kwa family proxy for Ga)")
    print(f"MODE         : {args.mode}")
    print(f"DATASET      : {args.dataset}")
    print(f"OUTPUT       : {args.output_dir}")
    print(f"MAX STEPS    : {args.max_steps}")
    if args.mode == "nonstandard":
        print(f"SS MODEL     : {args.ss_model_dir}")
        print(f"NSS DATA     : {args.nss_data_dir or 'HuggingFace (original)'}")
    print("=" * 65)

    # Load processor and model
    processor, model = load_processor_and_model(
        lang=args.lang,
        mode=args.mode,
        ss_model_dir=args.ss_model_dir,
    )

    # Load dataset
    if args.mode == "standard":
        train_ds, eval_ds, test_ds = load_hf_dataset(args.dataset, processor)
    elif args.nss_data_dir:
        train_ds, eval_ds, test_ds = load_augmented_nss(args.nss_data_dir, processor)
    else:
        train_ds, eval_ds, test_ds = load_hf_dataset(args.dataset, processor)

    print(f"\nFinal dataset sizes:")
    print(f"  Train : {len(train_ds):,}")
    print(f"  Eval  : {len(eval_ds):,}")
    if test_ds:
        print(f"  Test  : {len(test_ds):,}")

    # Build trainer components
    data_collator   = DataCollatorCTCWithPadding(processor=processor)
    compute_metrics = build_compute_metrics(processor)
    training_args   = get_training_args(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        mode=args.mode,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # NOTE: Trainer not Seq2SeqTrainer — CTC has no autoregressive generation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train
    print(f"\nStarting {args.mode} MMS training — {args.max_steps} steps...")
    if args.mode == "standard":
        print("  Trainable : ALL parameters (full fine-tune)")
        print("  SpecAug   : built into MMS wav2vec2 encoder")
        print("  Dataset   : standard Ghanaian speech (clean)")
    else:
        print("  Trainable : adapter + lm_head only (base encoder frozen)")
        print("  Base      : SS checkpoint (Ga phonology preserved)")
        if args.nss_data_dir:
            print(f"  Data      : augmented NSS ({len(train_ds):,} examples, 7× with MUSAN noise)")

    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save best model + processor
    best_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_path)
    processor.save_pretrained(best_path)
    print(f"\nBest model saved: {best_path}")

    # Final evaluation
    print("\n" + "=" * 65)
    print("FINAL EVALUATION")
    print("=" * 65)

    eval_sets = {"validation": eval_ds}
    if test_ds:
        eval_sets["test"] = test_ds

    for split_name, split_ds in eval_sets.items():
        r = trainer.evaluate(eval_dataset=split_ds)
        print(f"[{split_name.upper():>12}]  "
              f"WER={r.get('eval_wer', float('nan')):.4f}  "
              f"CER={r.get('eval_cer', float('nan')):.4f}  "
              f"Loss={r.get('eval_loss', float('nan')):.4f}")

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best model : {best_path}")

    if args.mode == "standard":
        print(f"\n  Next step — NSS fine-tuning:")
        print(f"  python3 train_mms.py \\")
        print(f"      --lang {args.lang} \\")
        print(f"      --mode nonstandard \\")
        print(f"      --ss_model_dir {best_path} \\")
        print(f"      --dataset {DS_NONSTANDARD} \\")
        print(f"      --output_dir train_mms_{args.lang}_nss \\")
        print(f"      --max_steps 2000")
        print(f"\n  Or with augmentation:")
        print(f"  python3 train_mms.py \\")
        print(f"      --lang {args.lang} \\")
        print(f"      --mode nonstandard \\")
        print(f"      --ss_model_dir {best_path} \\")
        print(f"      --nss_data_dir /path/to/augmented_ga_nss_musan \\")
        print(f"      --output_dir train_mms_{args.lang}_nss_augmented \\")
        print(f"      --max_steps 2000")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
