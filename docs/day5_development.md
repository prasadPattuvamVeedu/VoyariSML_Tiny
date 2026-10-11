# VoyariLM Tiny — Day 5 / Stage 1: Correct grounded SFT data

This patch **does not train** or change existing `.pt` checkpoints. It adds CPU-only
scripts for data generation, validation, and a read-only evaluation runner.

## Why

Day 4 data always required `origin` and `duration_days`, forbade `destination`, and
used budgets starting at ₹6,500. Day 5 teaches independent origin/destination,
optional fields, small budgets, corrections, and tool calls. Existing SFT model
and tokenizer remain unchanged.

## Included files

- `scripts/day5_schema.py` — action contract + explicit label grounding checks.
- `scripts/generate_day5_training.py` — deterministic 4,000-conversation synthesis.
- `scripts/validate_day5_training.py` — checks schema and held-out prompt exclusion.
- `scripts/test_day5.py` — fast standard-library unit tests.
- `scripts/eval_day5.py` — read-only checkpoint dev evaluation (PyTorch; Kaggle recommended).
- `eval/day5_dev_v1.jsonl` — 24 *development* questions (never train on these).

## Step 1 — Run fast checks (no GPU)

Place patch files into the root of your `VoyariSML_Tiny` repo; do not replace
old scripts. From the repo root:

```powershell
python .\scripts\test_day5.py
```

## Step 2 — Generate Day 5 data (no GPU)

```powershell
python .\scripts\generate_day5_training.py `
  --output .\artifacts\data\sft_day5\day5_grounded_train.jsonl `
  --manifest .\artifacts\data\sft_day5\day5_manifest.json `
  --holdout .\eval\day4_diagnostic_v1.jsonl `
  --holdout .\eval\day5_dev_v1.jsonl `
  --count 4000

python .\scripts\validate_day5_training.py `
  --training-file .\artifacts\data\sft_day5\day5_grounded_train.jsonl `
  --holdout .\eval\day4_diagnostic_v1.jsonl `
  --holdout .\eval\day5_dev_v1.jsonl `
  --expected-count 4000 `
  --report .\artifacts\data\sft_day5\day5_validation.json
```

If your private Day 3 reviewed training data is available, also add
`--existing-train <path-to-mixed_train_reviewed_candidate.jsonl>` to **both**
commands. Add older independent development holdouts similarly.

Output `day5_grounded_train.jsonl` is an instruction-tuning **candidate**,
not a validated model. Human-review samples of every action category, negative
examples, field roles and multi-turn cases before training.

## Step 3 — Baseline with *exact* Kaggle tokenizer and saved step75

On a Kaggle environment with PyTorch, tokenizers, the repo, Day4 Step75 `.pt`,
and its matching original tokenizer (validated by SHA256):

```bash
python scripts/eval_day5.py \
  --checkpoint /path/to/voyari_day4_step_000075.pt \
  --tokenizer /path/to/original/voyari_tokenizer_16k_v3.json \
  --eval-file eval/day5_dev_v1.jsonl \
  --output-dir /kaggle/working/day5_baseline
```

The evaluator uses the exact original training system text already shared by
user (or `--system-dataset /path/to/01_voyari_v9_train.jsonl`) and the existing
Day 4 Kaggle greedy evaluator's formatting. It **does not execute** tool calls.

## Future Stage 2/3 (not part of this patch)

- Review generated labels for non-literal ambiguity and repeated templates.
- Build a new `train_sft_day5.py` explicitly accepting Step75 as an immutable
  parent (do not reuse Day4 trainer; its parent validator requires SFT v2).
- Group/hold out related prompt templates and conversations to reduce train/val
  leakage; preserve existing Day3/Day4 replay data.
- Train 25 updates first, then evaluate; promote only on held-out accuracy.
- Django must independently validate state updates, tool allowlisting and user
  permissions. The model's JSON is a proposal, not an authorized command.
- `search_travel_packages`, `search_hotels`, `search_transport`,
  `search_activities` are **proposed** tool contracts, not live integrations.

## No-overwrite rules

- Do not commit `.pt` files, private datasets or user information.
- Do not rename or overwrite the Step75/100 checkpoint.
- Keep `eval/day4_diagnostic_v1.jsonl` and `eval/day5_dev_v1.jsonl` strictly out
  of all training input files.
