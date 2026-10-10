# Day 4 — Correct slot extraction and tool arguments

Day 3 baseline on the iterated 18-question diagnostic: SFT v2 step 100 achieved
17/18 valid JSON, 10/18 correct action type, 1/6 exact tool arguments,
0/6 required state fields. Day 3 reviewed data contains 960 conversations
and 1,447 assistant responses, with zero warnings under the current heuristic audit.
Keep the original SFT v1 and SFT v2 checkpoint files immutable.

## Day 4 diagnostic (new, frozen development set)

- `eval/day4_diagnostic_v1.jsonl`: 30 new questions, 12 state updates,
  8 tools, 10 clarification cases, with explicit gold fields.
- `scripts/day4_diagnostic.py`: integrity and exact-prompt overlap checks,
  plus model inference and stricter scoring when invoked without `--check-only`.
- `scripts/test_day4_diagnostic.py`: six fast scoring regression checks.
- The first notebook cell should pull the GitHub branch, execute the six tests,
  validate the 30 JSONL cases against the reviewed 960-conversation dataset,
  and report where the saved checkpoint can be found.
- Do not include this eval JSONL in any SFT training command. Never tune
  repeatedly until the benchmark is memorized.
- This benchmark was created in response to known model problems and is a
  **development diagnostic**, not a final unbiased blind-test claim.
  For a later release, set aside a separate untouched evaluation set.

## Step 1 (no GPU training)

```python
import subprocess, sys
from pathlib import Path

repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
data_name = "mixed_train_reviewed_candidate.jsonl"
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
subprocess.run([sys.executable, str(repo / "scripts/test_day4_diagnostic.py")],
               cwd=repo, check=True)

locations = [
    Path("/kaggle/working/VoyariSML_Tiny/artifacts/data/sft_v2"),
    Path("/kaggle/input"),
]
data = next(
    (p for base in locations for p in base.rglob(data_name) if p.is_file()),
    None,
)
if data is None:
    raise FileNotFoundError(
        "Attach the private Kaggle backup dataset, or restore the reviewed data."
    )

subprocess.run([
    sys.executable, "-u", str(repo / "scripts/day4_diagnostic.py"),
    "--check-only",
    "--training-file", str(data),
], cwd=repo, check=True)

checkpoint_candidates = [
    *Path("/kaggle/input").rglob("Voyari_SFT_v2_step_000100.pt"),
    *Path("/kaggle/working").rglob("voyari_sft_step_000100.pt"),
    *Path("/kaggle/working").rglob("Voyari_SFT_v2_step_000100.pt"),
]
print("Model checkpoint candidates:", [str(p) for p in checkpoint_candidates if p.is_file()])
```

The next step is to run Day 4 inference against the saved SFT v2 checkpoint,
inspect per-case failures, then prepare *new* validated state/tool training
examples without leaking the 30 development questions into training.


## Step 3 — Grounded new training data (no optimizer steps)

The first 12 Day 4 state cases showed 5 wrong actions, 5 incorrect origins
among the 7 state responses, 4 incorrect budgets, 3 incorrect durations,
and invented state fields. The new generator focuses on these observed failure
classes without copying benchmark prompts.

- `scripts/generate_day4_grounded_sft.py` builds 1,640 deterministic synthetic
  conversations (default: 1,200 grounded state updates, 240 clarification
  cases, 200 tool calls) using the exact existing V9 system instruction.
- `scripts/validate_day4_grounded_sft.py` checks role/JSON shape, implied
  group size, absence of invented fields, current label heuristics, and
  exact prompt exclusion against the reviewed Day 3 data and both development
  evaluations.
- Neither script trains or overwrites Day 3 checkpoints/data. New JSONL and
  manifest are stored in Kaggle output, not committed to GitHub.
- **Do not use** `eval/day4_diagnostic_v1.jsonl` as training material. Avoid
  repeated optimization directly on its exact prompts.
- All generated labels need spot review: deterministic template generation is
  useful for supervision but does not guarantee generalization.

```python
import subprocess, sys
from pathlib import Path

repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
v1_dir = Path(
    "/kaggle/input/datasets/prasadpattuvamveedu/voyari-tiny-instruction-v1"
)
v1_train = v1_dir / "01_voyari_v9_train.jsonl"
sgd_train = v1_dir / "02_sgd_travel_clarification_train.jsonl"
data = Path("/kaggle/working/VoyariSML_Tiny/artifacts/data/sft_v2")
reviewed = data / "mixed_train_reviewed_candidate.jsonl"
eval_old = data / "behavior_eval_independent.jsonl"
eval_new = repo / "eval/day4_diagnostic_v1.jsonl"
out = Path("/kaggle/working/day4_data")
out.mkdir(parents=True, exist_ok=True)
train = out / "day4_grounded_train.jsonl"

inputs = [v1_train, sgd_train, reviewed, eval_old, eval_new]
for f in inputs:
    assert f.is_file(), f"Missing required dataset: {f}"

base = [
    "--system-dataset", str(v1_train),
    "--holdout", str(eval_old), "--holdout", str(eval_new),
    "--existing-train", str(reviewed),
    "--existing-train", str(v1_train),
    "--existing-train", str(sgd_train),
]
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/generate_day4_grounded_sft.py"),
    "--output", str(train),
    "--manifest", str(out / "day4_grounded_manifest.json"),
    *base,
], cwd=repo, check=True)
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/validate_day4_grounded_sft.py"),
    "--training-file", str(train),
    "--holdout", str(eval_old), "--holdout", str(eval_new),
    "--existing-train", str(reviewed),
    "--existing-train", str(v1_train),
    "--existing-train", str(sgd_train),
    "--expected-count", "1640",
], cwd=repo, check=True)
```

Review printed counts and spot-check generated examples before deciding whether
to mix the new data with Day 3 replay data for a new, separate SFT experiment.


### Day 4 generator fix (2026-10-10)

The first generator run failed with `ValueError: Input/output paths must be distinct`
because the original V9 dataset was deliberately passed as both
`--system-dataset` and `--existing-train`. Repeating an input file is valid.
The generator now forbids only output/manifest collisions with each other or
with input files. Clarification templates also include location variation so
the default 240 unique clarification examples can be produced.

Before rerunning the generation command above, pull the latest branch and run:

```python
import subprocess, sys
from pathlib import Path
repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
subprocess.run(
    [sys.executable, "-u", str(repo / "scripts/test_day4_grounded_generation.py")],
    cwd=repo, check=True,
)
```

This is a code/path validation fix; it does not resume training or change model weights.


## Step 4 — Independent label audit and human spot review

After the 1,640 generated examples pass \`validate_day4_grounded_sft.py\`,
run the separate \`review_day4_grounded_labels.py\` checker. It compares
state numbers, interests, cities, budgets and budget bases against source text
and verifies tool argument and clarification targets. The script reports
warnings but **does not edit input training data or train the model**.

\`\`\`python
import subprocess, sys
from pathlib import Path

repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
subprocess.run(
    [sys.executable, "-u", str(repo / "scripts/test_review_day4_grounded_labels.py")],
    cwd=repo, check=True,
)
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/review_day4_grounded_labels.py"),
    "--train", "/kaggle/working/day4_data/day4_grounded_train.jsonl",
    "--report", "/kaggle/working/day4_data/day4_independent_review.json",
    "--sample-state", "8", "--sample-tool", "4", "--sample-clarify", "4",
], cwd=repo, check=True)
\`\`\`

Inspect the 16 cases printed, including **USER** and **EXPECTED** JSON.
A passing heuristic check alone is not enough for production training labels.
Record any semantic mismatches and correct the generator, not the Kaggle JSONL
by hand, so generated examples remain reproducible.


## Step 5 — Day 4 controlled SFT training preflight (NO GPU updates)

Use \`scripts/train_sft_day4.py\`, **not** \`train_sft_v2.py\`, so the
model starts from the frozen SFT v2 step-100 checkpoint instead of SFT v1
step 8000.

- Parent model: \`artifacts/checkpoints/sft_v2/corrective_v1/voyari_sft_step_000100.pt\`
- Fresh optimizer, new stage: \`sft_day4\`
- Run name: \`grounded_mix_v1\`
- Replay mix: 1,640 new synthetic examples + 960 reviewed Day 3 conversations
  (dataset index/shuffle combines them; no evaluation cases in training)
- Set \`--max-steps 100 --session-steps 25\`: only 25 updates in the first
  session, with a saved checkpoint and validation loss at step 25.
- New model output path:
  \`artifacts/checkpoints/sft_day4/grounded_mix_v1/voyari_day4_step_000025.pt\`
- Initial learning rate experiment: peak 3e-6, min 8e-7, 8 warmup steps,
  8 microbatches/optimizer update, fresh AdamW, evaluation every 25 steps.
- Mandatory holdout files: old 18-question development eval and Day 4
  30-question diagnostic. The trainer checks exact prompt overlap.
- Compare checkpoint with the frozen SFT v2 model before extending 25 steps.
  The Day 4 evaluator now accepts \`sft_day4\` staged checkpoints.

Run preflight first:

\`\`\`python
import subprocess, sys
from pathlib import Path
repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
subprocess.run(
    [sys.executable, str(repo / "scripts/test_train_sft_day4.py")],
    cwd=repo, check=True,
)
checkpoint = repo / "artifacts/checkpoints/sft_v2/corrective_v1/voyari_sft_step_000100.pt"
day3 = Path("/kaggle/working/VoyariSML_Tiny/artifacts/data/sft_v2")
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/train_sft_day4.py"),
    "--dry-run",
    "--base-checkpoint", str(checkpoint),
    "--training-file", "/kaggle/working/day4_data/day4_grounded_train.jsonl",
    "--training-file", str(day3 / "mixed_train_reviewed_candidate.jsonl"),
    "--holdout-file", str(day3 / "behavior_eval_independent.jsonl"),
    "--holdout-file", str(repo / "eval/day4_diagnostic_v1.jsonl"),
    "--max-steps", "100", "--session-steps", "25",
], cwd=repo, check=True)
\`\`\`

A passed preflight confirms data and checkpoint compatibility but performs
**no optimizer updates**. After the first 25 updates, run the same 30-question
benchmark for the frozen parent and the candidate. Save important checkpoints
to a persistent private Kaggle Dataset before notebook shutdown.


## Step 6 — Back up the new Day 4 step-25 candidate before any more training

The first 25 optimizer updates completed successfully, but the Day 4
development benchmark found no net improvement: exact total remained 4/30;
tool exact improved 2/8 → 4/8, but clarification topic match fell 2/10 → 0/10
and exact state remained 0/12. **Do not blindly resume to step 50.**

Use \`scripts/backup_day4_checkpoint.py\` to create a NEW private Kaggle
Dataset of the current Day 4 candidate checkpoint, generated training JSONL,
generator manifest, test results/summary and training log. The script validates
the step-25 parent lineage, copies files without modification, uses license
\`other\` (already accepted by Kaggle), and checks the Kaggle remote file listing.
No \`--public\` flag is used. Verify private visibility in Kaggle UI.

\`\`\`python
import subprocess, sys
from pathlib import Path
repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run([
    "git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"
], check=True)
checkpoint = repo / "artifacts/checkpoints/sft_day4/grounded_mix_v1/voyari_day4_step_000025.pt"
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/backup_day4_checkpoint.py"),
    "--checkpoint", str(checkpoint),
    "--train", "/kaggle/working/day4_data/day4_grounded_train.jsonl",
    "--train-manifest", "/kaggle/working/day4_data/day4_grounded_manifest.json",
    "--eval-results", "/kaggle/working/day4_evaluation/day4_sft_day4_step_000025_results.jsonl",
    "--eval-summary", "/kaggle/working/day4_evaluation/day4_sft_day4_step_000025_summary.json",
    "--log", str(repo / "logs/sft_day4/grounded_mix_v1.jsonl"),
], cwd=repo, check=True)
\`\`\`

After the private backup is verified, inspect the full 30-case action regressions
and missing/extra state fields; design a controlled follow-up rather than
continuing the same 100-step run automatically.


### If Kaggle reports "Your private Dataset is being created" followed by a failed file listing

Kaggle dataset creation may continue server-side after the CLI has uploaded
all files. The first \`kaggle datasets files\` attempt can fail while the new
dataset is indexed. **Do not rerun the full backup command**, as that can
attempt duplicate dataset creation.

The backup script now retries listing, and supports a verification-only mode.
To verify an existing dataset without reuploading a single file:

\`\`\`python
import subprocess, sys
from pathlib import Path
repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run([
    "git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"
], check=True)
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/test_backup_day4_checkpoint.py")
], cwd=repo, check=True)
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/backup_day4_checkpoint.py"),
    "--verify-only",
], cwd=repo, check=True)
\`\`\`

Verification uses the local backup manifest to check all seven expected
uploaded artifacts. A completed file listing confirms remote presence; for
privacy also confirm the Kaggle dataset visibility is set to **Private**.


## Step 7 — Analyze Day 4 regression (no more training)

Day 4 step-25 backup upload and the seven file listings were verified in
Kaggle private Dataset:
\`prasadpattuvamveedu/voyari-tiny-day4-step25-backup-20261010\`.
Confirm Kaggle visibility is set to Private.

Current development benchmark comparison:
- SFT v2 step100: valid JSON 29/30, correct action 19/30, exact total 4/30;
  state exact 0/12, tool exact 2/8, clarification topic 2/10.
- Day 4 step25: valid JSON 30/30, correct action 19/30, exact total 4/30;
  state exact 0/12, tool exact 4/8, clarification topic 0/10.
- Clarification *action selection* regressed from 8/10 to 2/10. This is more
  substantial than the keyword-based topic score. Investigate before training.

\`scripts/compare_day4_checkpoints.py\` prints per-question before/after
responses and action transitions, focusing on clarification failures first.
\`scripts/test_compare_day4_checkpoints.py\` tests comparison invariants.

\`\`\`python
import subprocess, sys
from pathlib import Path
repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
subprocess.run(
    [sys.executable, str(repo / "scripts/test_compare_day4_checkpoints.py")],
    cwd=repo, check=True,
)
folder = Path("/kaggle/working/day4_evaluation")
subprocess.run([
    sys.executable, "-u", str(repo / "scripts/compare_day4_checkpoints.py"),
    "--baseline", str(folder / "day4_sft_v2_step_000100_results.jsonl"),
    "--candidate", str(folder / "day4_sft_day4_step_000025_results.jsonl"),
    "--baseline-summary", str(folder / "day4_sft_v2_step_000100_summary.json"),
    "--candidate-summary", str(folder / "day4_sft_day4_step_000025_summary.json"),
    "--category", "clarify", "--limit", "10",
    "--report", str(folder / "day4_regression_comparison.json"),
], cwd=repo, check=True)
\`\`\`

Inspect the complete text of all ten original clarification questions and
both model responses. The keyword topic metric can misjudge a response, but
wrong action types are separately counted. Use this analysis to guide data
changes, rather than extending the existing step-25 training run blindly.
