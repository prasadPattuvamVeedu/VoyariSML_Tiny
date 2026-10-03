# VoyariLM Tiny - Stage 1 Pretraining

## Purpose

This is the first real training stage after the 5-step/6-step smoke tests.

The goal is **stability**, not fluency yet:

- verify the full 14-layer model can train continuously;
- verify tokenizer v3 and final pretraining corpora are used;
- verify loss and gradient norms remain finite;
- verify mixed precision works on CUDA;
- verify checkpoint save/resume works;
- collect a clean metrics log before a longer Kaggle run.

## Current model

- Vocabulary: 16,000
- Context: 1,024
- Hidden size: 384
- Layers: 14
- Heads: 6
- Head dimension: 64
- FFN: 1,024
- Tied token embedding / LM head
- RoPE positional encoding
- Approximate trainable parameters: 30.93M

With a 16,000-token vocabulary, a completely untrained model has a reference cross-entropy near ln(16000), about **9.68**. Individual steps can move up and down. For Stage 1, look for a generally improving trend, not a perfectly decreasing loss every step.

## What changed before Stage 1

The training branch fixes two blockers in the old smoke-test pipeline:

1. The shared data config now points to tokenizer v3.
2. Training data paths are repository-relative instead of hard-coded to `D:\voyari_sml`.

The loader also refuses to train when a corpus is still only a Git LFS pointer.

## Windows run

From the repository root:

```powershell
cd D:\voyari_sml

git fetch origin
git checkout training/stage1-stability

git lfs pull

python -m src.data.pretraining_formatter
python -m src.data.pretraining_sequences

python scripts/pretrain_stage1.py --max-steps 200 --batch-size 1 --grad-accum 8
```

## Kaggle run

Make sure the repository and the actual Git LFS corpus files are present in the Kaggle working environment. Then run from the repository root:

```bash
python scripts/pretrain_stage1.py \
  --max-steps 200 \
  --batch-size 1 \
  --grad-accum 8 \
  --save-every 50 \
  --log-every 10
```

CUDA uses bf16 when supported; otherwise fp16 with gradient scaling. CPU falls back to fp32.

## Resume

Stage 1 automatically finds the highest numbered checkpoint in:

```text
artifacts/checkpoints/stage1/
```

Run the same command again to continue.

To force a fresh Stage-1 run:

```bash
python scripts/pretrain_stage1.py --resume none
```

To select a specific checkpoint:

```bash
python scripts/pretrain_stage1.py --resume artifacts/checkpoints/stage1/voyari_stage1_step_000100.pt
```

## Outputs

Checkpoints:

```text
artifacts/checkpoints/stage1/voyari_stage1_step_000050.pt
artifacts/checkpoints/stage1/voyari_stage1_step_000100.pt
...
```

Metrics:

```text
logs/stage1_metrics.jsonl
```

Each metric record contains optimizer step, loss, gradient norm, learning rate, tokens seen, tokens/second, and peak CUDA memory.

## Stage-1 acceptance check

Proceed to the longer pretraining stage only when:

- no NaN/Inf loss occurs;
- gradient norm remains finite;
- checkpoints can resume correctly;
- loss shows a meaningful downward trend across the run;
- GPU memory remains stable.

Do not use the old `voyari_step_5.pt` smoke-test checkpoint as the real training base. Stage 1 starts the actual run cleanly and keeps its checkpoints in a separate folder.
