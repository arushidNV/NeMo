#!/usr/bin/env bash
# Full cache-aware RNNT streaming eval: stages the dataset locally, clones/updates
# the eval-streaming-inference branch, pulls the eval container, and runs all
# 4 profile x decoding combos (mono/multi x greedy/beam), writing results to CIFS.
#
# Run this directly on a leased dlcluster node (bare host, NOT inside Docker --
# this script launches Docker itself for each profile run).
#
# Usage:
#   ./run_full_eval.sh
#   DATASET_SRC=/some/other/path ./run_full_eval.sh   # override any var below

set -euo pipefail

DATASET_SRC="${DATASET_SRC:-/home/scratch.mayjain_sw/datasets/en}"
STAGE_DIR="${STAGE_DIR:-/tmp/eval_staging}"
REPO_DIR="${REPO_DIR:-/tmp/NeMo}"
REPO_URL="${REPO_URL:-https://github.com/arushidNV/NeMo.git}"
REPO_BRANCH="${REPO_BRANCH:-eval-streaming-inference}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/nemo-speech:26.07.00}"
CIFS_OUT_BASE="${CIFS_OUT_BASE:-/mnt/cifs/home/riva_speech/mayjain/arushid}"
CIFS_MODEL_BASE="${CIFS_MODEL_BASE:-/mnt/cifs/home/riva_speech/model_files}"
MONO_MODEL="$CIFS_MODEL_BASE/asr/cache-aware-parakeet-rnnt/en-US/nemotron_realtime_0.6b_riva.nemo"
MULTI_MODEL="$CIFS_MODEL_BASE/asr/cache-aware-parakeet-rnnt/multi/nemotron_realtime_0.6b_multi.nemo"

echo "=== 1. Verify /tmp is local disk, not tmpfs ==="
STAGE_PARENT_FS=$(df --output=fstype "$(dirname "$STAGE_DIR")" | tail -1 | tr -d ' ')
if [ "$STAGE_PARENT_FS" = "tmpfs" ]; then
    echo "ERROR: $(dirname "$STAGE_DIR") is tmpfs (RAM-backed) on this node. Refusing to stage a large" >&2
    echo "dataset there -- set STAGE_DIR to a real local disk path instead." >&2
    exit 1
fi
echo "OK: $(dirname "$STAGE_DIR") is $STAGE_PARENT_FS"

echo "=== 2. Stage dataset from $DATASET_SRC to $STAGE_DIR (skip if already staged) ==="
if [ -d "$STAGE_DIR" ] && [ -n "$(ls -A "$STAGE_DIR" 2>/dev/null)" ]; then
    echo "Skipping: $STAGE_DIR already exists and is non-empty."
else
    mkdir -p "$STAGE_DIR"
    time tar -C "$DATASET_SRC" -cf - . | tar -C "$STAGE_DIR" -xf -
fi

echo "=== 3. Verify model checkpoints are reachable ==="
for f in "$MONO_MODEL" "$MULTI_MODEL"; do
    if [ ! -r "$f" ]; then
        echo "ERROR: model checkpoint not readable: $f" >&2
        exit 1
    fi
    echo "OK: $f"
done

echo "=== 4. Clone or update the eval branch at $REPO_DIR ==="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull origin "$REPO_BRANCH"
else
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
fi

echo "=== 5. Pull the eval container ==="
docker pull "$DOCKER_IMAGE"

echo "=== 6. Run all 4 profile x decoding combos ==="
mkdir -p "$CIFS_OUT_BASE"

for combo in "mono greedy" "mono beam" "multi greedy" "multi beam"; do
    set -- $combo
    profile="$1"
    decoding="$2"
    out_name="${profile}_${decoding}"
    echo "--- Running profile=$profile decoding=$decoding -> $CIFS_OUT_BASE/$out_name ---"
    docker run --rm --gpus all --shm-size=16g \
        -v "$REPO_DIR:/workspace/NeMo" \
        -v "$STAGE_DIR:/workspace/data:ro" \
        -v "/mnt/cifs/home/riva_speech:/mnt/cifs/home/riva_speech" \
        "$DOCKER_IMAGE" bash -c "
            cd /workspace/NeMo && \
            echo y | python3 examples/asr/asr_streaming_inference/comm_streaming_nemo.py \
                '$CIFS_OUT_BASE/$out_name' \
                --profile '$profile' --decoding '$decoding' \
                --manifest_dir /workspace/data
        "
done

echo "=== Done. Results under $CIFS_OUT_BASE/{mono,multi}_{greedy,beam}/ ==="
echo "Remember to clean up node-local scratch when done with this lease:"
echo "  rm -rf $STAGE_DIR $REPO_DIR"
