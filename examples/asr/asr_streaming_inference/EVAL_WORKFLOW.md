# Cache-aware streaming ASR eval workflow

Everything below lives on branch `eval-streaming-inference` at
`git clone --branch eval-streaming-inference https://github.com/arushidNV/NeMo.git`.
That's the only thing you need to hand someone to let them reproduce all of this.

Scripts: `comm_streaming_nemo.py` (WER eval, per-manifest), `run_full_eval.sh` (orchestrates
staging + docker + comm_streaming_nemo.py across profiles), `asr_streaming_infer.py` (the
underlying NeMo inference entry point, called by both). Config profiles:
`../conf/asr_streaming_inference/cache_aware_rnnt_{mono,multi}_{greedy,beam}.yaml`.

## 0. Requirements

- A CUDA GPU + Docker with `nvidia-container-toolkit`. Sanity check:
  `nvidia-smi` and `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
- Docker image: `nvcr.io/nvidia/nemo-speech:26.07.00` (pulled automatically by `run_full_eval.sh`).
- If running on NVIDIA's dlcluster (Slurm), see "Leasing a dlcluster node" below.

## 1. Leasing a dlcluster node (skip if you already have a GPU box)

```bash
sinfo | grep idle
```
Filter for x86_64 nodes tagged `santa_clara` (or whichever site is closest/fastest for you) —
cross-reference against the fuller feature dump if `sinfo | grep idle` alone doesn't show site/arch:
```bash
sinfo -N -o "%N %f %l %P" | grep -i "santa_clara" | grep -v "aarch64"
```
Prefer H100 nodes (`s4124-*`, `ipp2-*` HBM3/HBM2e variants have shown up reliably idle) unless you
need multi-GPU. Once you've picked a node, from a **fresh terminal in a persistent session (tmux, or
plain nohup+disown if tmux isn't behaving)**:

```bash
git clone --branch eval-streaming-inference https://github.com/arushidNV/NeMo.git /tmp/NeMo
```

If you're not inside a real tmux session and can't rely on the terminal staying open, run inside
`srun` directly (not `--pty`) and detach the shell job instead of relying on tmux — we hit repeated,
unexplained failures trying to `tmux new-session -d ...` with complex multi-line commands pasted at
once on this cluster's login nodes. What reliably worked:
```bash
srun --nodelist=<NODE> --partition=general --time=8:00:00 bash -c '<your full command>' &
# then, in the same shell:
Ctrl+Z
bg
disown -h
```
`disown -h` stops the shell from sending SIGHUP to the job on exit, so it survives closing the
terminal even without tmux.

## 2. Full WER eval over a dataset

One-shot, all 4 profile×decoding combos (mono/multi × greedy/beam), staging the dataset from CIFS
to local disk once and reusing it across all 4:

```bash
/tmp/NeMo/examples/asr/asr_streaming_inference/run_full_eval.sh
```

Key env vars (all optional, see the top of the script for full defaults):

| Var | Purpose |
|---|---|
| `DATASET_SRC` | Source manifest+audio directory (default: a CIFS path) |
| `COMBOS` | Restrict to specific profile:decoding pairs, e.g. `COMBOS="mono:beam"` or `COMBOS="mono:beam;multi:beam"` |
| `OUT_SUFFIX` | Append to the output folder name (e.g. `_lm`, `_lnp0`) so variant runs don't collide |
| `EXTRA_ARGS` | Passed straight through to `comm_streaming_nemo.py` — see flags below |
| `MONO_MODEL` / `MULTI_MODEL` | Override which checkpoint path gets verified in step 3 (keep in sync with `--model_name` in `EXTRA_ARGS` if you're using a non-default checkpoint) |
| `CIFS_OUT_BASE` | Where results land (default a CIFS path) |

What it actually does, in order: (1) checks `/tmp` isn't RAM-backed, (2) stages the dataset to local
disk — **skips if already staged from a prior run on this node**, (3) verifies the checkpoint(s)
needed for the requested `COMBOS`, (4) clones/pulls the branch, (5) pulls the docker image, (6) runs
each combo, mounting the repo + staged data + CIFS + `/home` (read-only) into the container.

`comm_streaming_nemo.py` flags available via `EXTRA_ARGS`:

```
--model_name PATH           # override asr.model_name (e.g. a personal scratch checkpoint)
--batch_size N               # override streaming.batch_size
--length_norm_power N        # beam only; 1.0=default, 0.0=off (see beam-degeneration note below)
--ngram_lm_model PATH        # fuse a built NGPU-LM .nemo (beam only)
--ngram_lm_alpha N           # LM fusion weight (NeMo docs suggest ~0.2 for RNNT)
--datasets ds1,ds2,...       # restrict to specific manifest names instead of every *.json
--reverse                    # process manifests in reverse order
```

Output layout per combo: `<out_dir>/<dataset_name>.json` (pred_text + per-utterance metrics),
`<out_dir>/segments/` (word timing detail, usually skippable), `<out_dir>/wer.csv`.

## 3. Word-boosting sweeps

Different flow — invoke `asr_streaming_infer.py` directly (not through `comm_streaming_nemo.py`,
which doesn't expose boosting flags), once per score value you want to sweep:

```bash
docker run --rm --gpus all --shm-size=8g \
    -v <repo>:/workspace/NeMo \
    -v <audio_dir>:/workspace/wav:ro \
    -v <model_dir>:/mnt/model:ro \
    -v <manifests_dir>:/workspace/manifests \
    -v <out_dir>:/workspace/out \
    nvcr.io/nvidia/nemo-speech:26.07.00 bash -c '
        export PYTHONPATH="/workspace/NeMo:${PYTHONPATH:-}"
        python3 /workspace/NeMo/examples/asr/asr_streaming_inference/asr_streaming_infer.py \
            --config-path /workspace/NeMo/examples/asr/conf/asr_streaming_inference \
            --config-name cache_aware_rnnt_mono_beam.yaml \
            asr.model_name=/mnt/model/<checkpoint>.nemo \
            audio_file=/workspace/manifests/<case>.json \
            output_filename=/workspace/out/<case>_alpha<N>.json \
            output_dir=/workspace/out/segments_<case>_<N> \
            calculate_wer=false \
            "asr.decoding.beam.boosting_tree.key_phrases_list=[\"<target phrase>\"]" \
            asr.decoding.beam.boosting_tree_alpha=<N>
    '
```

Read `json.loads(open(output_filename).readline())["pred_text"]`. Multi-word phrases and phrases
with apostrophes need double-quoting *inside* the list brackets (`["Mrs Bectors'"]`), not bare.

**Don't just check "did the target word appear."** Diff each alpha step against the previous step.
Two real failure modes to watch for: (a) boosting can flip every phonetically-matching span in the
transcript, not just the intended one; (b) high alpha can trigger a short repetition loop of the
boosted token before the decoder recovers.

## 4. Known gotchas

- **`nemo` import collision**: the prebuilt container ships its own baked-in `nemo` package, older
  than this branch, missing symbols the scripts expect. Always
  `export PYTHONPATH="/workspace/NeMo:${PYTHONPATH:-}"` before any `python3` invocation inside the
  container if you're not going through `comm_streaming_nemo.py` (which already sets this for its
  own subprocess calls).
- **Pipeline type matters**: check whether a model is `is_cache_aware: true/false` in its bundled
  `streaming_config.yaml`/`riva_config.json` (if it was ever deployed via Riva) before picking a
  config profile — cache-aware models need `cache_aware_rnnt*.yaml`, non-cache-aware ("buffered")
  models need `buffered_rnnt.yaml`/`buffered_ctc.yaml`, with different `chunk_size`,
  `left_padding_size`/`right_padding_size`, and `endpointing.*` values to match.
- **`calc_wer.py` field priority**: if scoring against a separate reference set via
  `/opt/riva/utils/calc_wer.py`, it prefers a `"text"` key over `"pred_text"` when loading the
  *test* file. Our own output carries the original ground-truth `"text"` through alongside
  `pred_text`, so scoring directly against our raw output silently compares ground truth to itself
  (impossible 0.00% WER everywhere). Strip `"text"` from a temp copy first.
- **Beam-search degeneration (`-inf` score underflow)**: on very long audio (40+ min single
  utterances), we found `hyp_decoding_state.score` in `cache_aware_rnnt_pipeline.py` is a running
  log-prob sum that's never actually reset — only baseline-subtracted for ranking. It can underflow
  to literal `-inf`, after which beam selection can't discriminate hypotheses and the decoder runs
  unsteered into repetition collapse (500%+ WER). `length_norm_power=0` sidesteps it on the files we
  tested by taking a different search path, but doesn't fix the root cause. To debug this on a new
  model, instrument `_apply_beam_update_` gated behind an env var (e.g. `NEMO_DEBUG_EOU_FOLD=1`,
  passed via `docker run -e`) to log score/length at every EOU fold.

## Reference: what we validated this against

WER eval across 33 datasets, mono (en-US) and multilingual cache-aware 0.6B models, both greedy and
beam decoding. Word-boosting sweeps on that 0.6B model and a real `parakeet-rnnt-1.1b` multilingual
buffered model (found both cross-occurrence over-application and repetition-loop behavior on the
0.6B model that the 1.1B model didn't show). LM fusion via `--ngram_lm_model`/`--ngram_lm_alpha`.
Beam-search `-inf` degeneration reproduced independently on two different machines with the same
code/model, confirming it wasn't a one-off fluke.
