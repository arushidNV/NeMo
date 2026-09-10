#!/usr/bin/env python3
"""
Runs the cache-aware RNNT streaming pipeline (asr_streaming_infer.py) over every
*.json manifest in a directory, for a chosen model profile (mono/multi) and
decoding mode (greedy/beam), then aggregates WER into a CSV.

Config profiles live in ../conf/asr_streaming_inference/:
    cache_aware_rnnt_mono_greedy.yaml   cache_aware_rnnt_mono_beam.yaml
    cache_aware_rnnt_multi_greedy.yaml  cache_aware_rnnt_multi_beam.yaml

Usage:
    python3 comm_streaming_nemo.py <out_dir> --profile {mono,multi} --decoding {greedy,beam}
                                    [--manifest_dir DIR] [--batch_size N] [--reverse]
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMING_INFER = os.path.join(SCRIPT_DIR, "asr_streaming_infer.py")
CONF_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "conf", "asr_streaming_inference")

PROFILE_CONFIG = {
    ("mono", "greedy"): "cache_aware_rnnt_mono_greedy.yaml",
    ("mono", "beam"): "cache_aware_rnnt_mono_beam.yaml",
    ("multi", "greedy"): "cache_aware_rnnt_multi_greedy.yaml",
    ("multi", "beam"): "cache_aware_rnnt_multi_beam.yaml",
}

try:
    from nemo.collections.asr.metrics.wer import word_error_rate
except ImportError as e:
    print(f"[warn] NeMo not importable for WER aggregation: {e}")
    word_error_rate = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Run NeMo cache-aware RNNT streaming inference over a directory of manifests."
    )
    p.add_argument("out_dir", help="Directory to write per-dataset transcripts + wer.csv to")
    p.add_argument("--profile", choices=["mono", "multi"], required=True, help="mono (en-US) or multi (multilingual)")
    p.add_argument("--decoding", choices=["greedy", "beam"], required=True, help="Decoding strategy")
    p.add_argument("--manifest_dir", default=".", help="Directory containing *.json manifests (default: cwd)")
    p.add_argument("--batch_size", type=int, default=None, help="Override streaming.batch_size (default: profile yaml default)")
    p.add_argument("--reverse", action="store_true", help="Process manifests in reverse sorted order")
    return p.parse_args()


def prepare_out_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        return
    answer = input(f"Output dir '{out_dir}' already exists. Overwrite (delete its contents)? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborting - out dir left untouched.")
        sys.exit(1)
    for entry in os.listdir(out_dir):
        path = os.path.join(out_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _ref_by_audio(manifest_path):
    ref_by_audio = {}
    with open(manifest_path, "r") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio = entry.get("audio_filepath")
            ref = entry.get("text")
            if audio is not None and ref is not None:
                ref_by_audio[audio] = ref
                ref_by_audio[os.path.basename(audio)] = ref
    return ref_by_audio


def compute_wer(manifest_path, out_json):
    """Join pred_text from out_json against text in the source manifest, mirroring the old
    comm_streaming_nemo.py convention (output schema doesn't carry ground truth reliably)."""
    if not os.path.exists(out_json) or os.path.getsize(out_json) == 0:
        return "SKIPPED"

    ref_by_audio = _ref_by_audio(manifest_path)
    hyps, refs = [], []
    with open(out_json, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            hyp = rec.get("pred_text")
            audio = rec.get("audio_filepath")
            if hyp is None or audio is None:
                continue
            ref = ref_by_audio.get(audio) or ref_by_audio.get(os.path.basename(audio))
            if ref is None:
                continue
            hyps.append(hyp)
            refs.append(ref)

    if not hyps:
        return "NO_DATA"
    if word_error_rate is None:
        return "NO_NEMO"
    return round(word_error_rate(hypotheses=hyps, references=refs) * 100, 2)


def main():
    args = parse_args()
    prepare_out_dir(args.out_dir)

    config_name = PROFILE_CONFIG[(args.profile, args.decoding)]

    manifests = sorted(
        (p for p in glob.glob(os.path.join(args.manifest_dir, "*.json")) if os.path.basename(p) != "final_transcripts.json"),
        reverse=args.reverse,
    )
    if not manifests:
        sys.exit(f"No *.json manifests found in {args.manifest_dir}")

    wer_results = []
    for manifest in manifests:
        name = os.path.basename(manifest)
        out_json = os.path.join(args.out_dir, name)
        out_segments = os.path.join(args.out_dir, "segments", os.path.splitext(name)[0])

        cmd = [
            sys.executable, STREAMING_INFER,
            "--config-path", CONF_DIR,
            "--config-name", config_name,
            f"audio_file={manifest}",
            f"output_filename={out_json}",
            f"output_dir={out_segments}",
            "calculate_wer=true",
        ]
        if args.batch_size is not None:
            cmd.append(f"streaming.batch_size={args.batch_size}")

        print(f"[run] {name} (profile={args.profile}, decoding={args.decoding})")
        subprocess.run(cmd)

        wer = compute_wer(manifest, out_json)
        print(f"{name}\t{wer}")
        wer_results.append((name, wer))

    csv_path = os.path.join(args.out_dir, "wer.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "wer"])
        writer.writerows(wer_results)
    print(f"\nWER results saved to {csv_path}")


if __name__ == "__main__":
    main()
