import io
import os
import csv
import random
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

SAMPLING_RATE = 16000
SEED          = 42
SNR_LEVELS    = [-10, -5, 0, 5, 10, 20]
OUTPUT_DIR    = "augmented_ga_nss_musan"

random.seed(SEED)
np.random.seed(SEED)

# ── Load MUSAN ────────────────────────────────────────────────────────────────
print("Loading MUSAN...")
musan = load_dataset("FluidInference/musan", split="train")
## tells hugging face to keep the audio as bytes and not decode it yet (we'll do it manually)
musan = musan.cast_column("audio", Audio(decode=False))
print(f"  Total clips: {len(musan)}")

all_noise = []
for ex in musan:
    try:
        ## load audio from bytes, convert to mono, resample if needed
        y, sr = sf.read(ex["audio"]["path"])
        if y.ndim > 1:
            y = y.mean(axis=1) ## stereo to mono by averaging channels
        if sr != SAMPLING_RATE:
            import librosa
            ## resample to target sampling rate
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLING_RATE)
        y = y.astype(np.float32)
        ## skip clips that are too short (less than 0.5 seconds) to ensure we have enough noise for augmentation
        if len(y) / SAMPLING_RATE >= 0.5:
            all_noise.append(y)
    except:
        pass

print(f"  Loaded {len(all_noise)} valid noise clips")

# Split noise per split — no overlap between splits
## we use 78% for train, 10% for val, and 12% for test (since test is usually smaller)
n           = len(all_noise)
train_noise = all_noise[:int(n * 0.78)]
val_noise   = all_noise[int(n * 0.78):int(n * 0.88)]
test_noise  = all_noise[int(n * 0.88):]
print(f"  Train noise: {len(train_noise)} | Val: {len(val_noise)} | Test: {len(test_noise)}")

# ── Load NSS ──────────────────────────────────────────────────────────────────
print("\nLoading NSS...")
nss = load_dataset("cdli/ghanian_ga_nonstandard_speech_v1.0")
nss = nss.cast_column("audio", Audio(decode=False))
for split in nss:
    print(f"  [{split}] {len(nss[split]):,} examples")

# ── SNR helpers ───────────────────────────────────────────────────────────────
## Compute RMS of a signal, with a small epsilon to avoid division by zero
## Measure the average power of the signal, which is needed to scale the noise to achieve the desired SNR
## its formula is sqrt(mean(x^2)) where x is the audio signal
## average power is mean(x^2), and RMS is the square root of that. We also ensure it's at least 1e-10 to avoid issues with very quiet signals.
def compute_rms(x):
    return float(np.sqrt(np.maximum(np.mean(x.astype(np.float64)**2), 1e-10)))

## SNR(dB) = 20 * log10(rms_signal / rms_noise)
## -10 dB means noise is 3x stronger than signal, 0 dB means equal power, +10 dB means signal is 3x stronger than noise, etc.
## To achieve a target SNR, we can scale the noise by a factor of (rms_signal / (10^(snr_db/20))) / rms_noise. This ensures that when we add the scaled noise to the signal, the resulting mixture has the desired SNR.
def mix_at_snr(speech, noise, snr_db):
    L = len(speech)
    ## If noise is shorter than speech, we loop it until it's long enough, then take a random segment of the appropriate length
    if len(noise) < L:
        ## loop repeat noise until it's long enough
        ##  int(np.ceil(L / len(noise))) calculates how many times we need to repeat the noise to ensure it's at least as long as the speech signal. We use np.tile to repeat the noise array that many times. This way, we can handle cases where the noise clip is shorter than the speech clip by effectively looping the noise.
        noise = np.tile(noise, int(np.ceil(L / len(noise))))
    ## take a random segment of the noise that matches the length of the speech
    offset = random.randint(0, max(0, len(noise) - L))
    ## we copy the segment to ensure it's contiguous in memory, which can help with performance when adding it to the speech signal
    noise  = noise[offset:offset+L].copy()
    ## compute RMS of speech and noise to determine how much to scale the noise to achieve the desired SNR
    rms_s  = compute_rms(speech)
    rms_n  = compute_rms(noise)
    if rms_n < 1e-8:
        return speech.copy()
    scale  = (rms_s / (10**(snr_db/20.0))) / rms_n
    ## We then add the scaled noise to the speech signal. To prevent clipping, we use np.clip to ensure that the resulting values are between -1.0 and 1.0, which is the typical range for floating-point audio signals. Finally, we convert the result to float32 for consistency.
    return np.clip(speech + noise * scale, -1.0, 1.0).astype(np.float32)


def snr_label(snr_db):
    return "clean" if snr_db == 999.0 else f"{snr_db:+.0f}dB"

# ── Columns ───────────────────────────────────────────────────────────────────
META_COLS = [
    "speaker_id", "prompt_type", "recording_environment",
    "recording_device", "confidence", "audio_length", "transcript_length",
]

CSV_COLS = [
    "aug_id", "split", "original_idx", "transcription",
    "speaker_id", "prompt_type", "recording_environment",
    "recording_device", "confidence", "audio_length", "transcript_length",
    "snr_db", "snr_label", "noise_type", "audio_filename",
]

# ── Augment one split ─────────────────────────────────────────────────────────
def augment_split(split_data, split_name, noise_pool, snr_levels, audio_out_dir):
    os.makedirs(audio_out_dir, exist_ok=True)
    rows  = []
    N     = len(split_data)

    print(f"\n  [{split_name}]")
    print(f"    Examples   : {N:,}")
    print(f"    SNR levels : {snr_levels}")
    print(f"    Noise pool : {len(noise_pool)} clips")
    print(f"    Output     : {N*(1+len(snr_levels)):,} examples")

    for i, ex in enumerate(split_data):
        if i % 500 == 0:
            print(f"    [{i:>5}/{N}] ...")

        # Load speech from bytes
        try:
            ## we read the audio from bytes using soundfile, convert to mono if needed, and resample to target sampling rate if needed
            speech, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
            if speech.ndim > 1:
                speech = speech.mean(axis=1)
            if sr != SAMPLING_RATE:
                import librosa
                speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLING_RATE)
            speech = speech.astype(np.float32)
        except Exception as e:
            print(f"    Skip {i}: {e}")
            continue

        tx   = ex.get("transcription", "")
        ## we create a base dictionary with the metadata for this example, which will be shared across all augmented versions (clean + noisy)
        meta = {c: ex.get(c) for c in META_COLS}
        base = {"split": split_name, "original_idx": i,
                "transcription": tx, **meta}

        # Save clean original
        fname_clean = f"{split_name}_{i:06d}_clean.wav"
        sf.write(os.path.join(audio_out_dir, fname_clean),
                 speech, SAMPLING_RATE, subtype="PCM_16")
        rows.append({**base,
                     "aug_id":         f"{split_name}_{i:06d}_clean",
                     "snr_db":         999.0,
                     "snr_label":      "clean",
                     "noise_type":     "clean",
                     "audio_filename": fname_clean})

        # Save noisy versions — all splits get full augmentation
        ## We randomly select a noise clip from the pool for each example, and then mix it with the speech at each SNR level. This means that different examples will have different noise clips, which adds more diversity to the augmented data.
        noise = random.choice(noise_pool)
        for snr_db in snr_levels:
            ## We mix the speech with the noise at the desired SNR level using the mix_at_snr function, which scales the noise to achieve the target SNR when added to the speech. The resulting noisy audio is then saved to disk with a filename that indicates the split, original index, and SNR level. We also create a row in our metadata CSV for each augmented example, which includes all the original metadata plus the augmentation details (SNR, noise type, etc.).
            noisy = mix_at_snr(speech, noise, snr_db)
            fname = f"{split_name}_{i:06d}_snr{snr_db:+.0f}.wav"
            ## we save the noisy audio to disk using soundfile, ensuring it's in 16-bit PCM format which is standard for WAV files. The filename includes the split name, original index, and SNR level for easy identification. We then append a new row to our metadata list with all the relevant information about this augmented example.
            sf.write(os.path.join(audio_out_dir, fname),
                     noisy, SAMPLING_RATE, subtype="PCM_16")
            rows.append({**base,
                         "aug_id":         f"{split_name}_{i:06d}_snr{snr_db:+.0f}",
                         "snr_db":         float(snr_db),
                         "snr_label":      snr_label(snr_db),
                         "noise_type":     "musan",
                         "audio_filename": fname})

    print(f"    Done → {len(rows):,} files saved to {audio_out_dir}/")
    return rows

# ── Run all splits ────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

split_noise_map = {
    "train":      train_noise,
    "validation": val_noise,
    "dev":        val_noise,
    "test":       test_noise,
}

all_csv_rows = []
for split in nss:
    noise_pool = split_noise_map.get(split, train_noise)
    audio_dir  = os.path.join(OUTPUT_DIR, "audio", split)
    rows       = augment_split(nss[split], split, noise_pool,
                               SNR_LEVELS, audio_dir)
    all_csv_rows.extend(rows)

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
    writer.writeheader()
    for row in all_csv_rows:
        writer.writerow({k: row.get(k, "") for k in CSV_COLS})

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DONE")
print("="*60)
for split in nss:
    split_rows = [r for r in all_csv_rows if r["split"] == split]
    clean      = sum(1 for r in split_rows if r["noise_type"] == "clean")
    noisy      = sum(1 for r in split_rows if r["noise_type"] == "musan")
    print(f"  [{split:>12}] {len(split_rows):>7,} total  "
          f"({clean:,} clean + {noisy:,} noisy)")

print(f"\n  Total rows : {len(all_csv_rows):,}")
print(f"  CSV        : {csv_path}")
print(f"\n  Output structure:")
print(f"    {OUTPUT_DIR}/")
print(f"    ├── metadata.csv")
print(f"    └── audio/")
for split in nss:
    n_files = len([r for r in all_csv_rows if r["split"] == split])
    print(f"        ├── {split}/   ({n_files:,} wav files)")
print(f"\n  Load CSV:")
print(f"    import pandas as pd")
print(f"    df = pd.read_csv('{csv_path}')")
print(f"    df[df.snr_db < 0]              # negative SNR only")
print(f"    df[df.split == 'train']        # train only")
print(f"    df[df.speaker_id == '787']     # by speaker")