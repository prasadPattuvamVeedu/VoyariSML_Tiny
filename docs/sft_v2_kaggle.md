# VoyariLM Tiny — corrective SFT v2 (Kaggle)

This run starts from **Day 2 SFT v1 step 8000**, not from Day 1 pretraining. It keeps v1 checkpoints untouched.

## Prerequisites

1. Kaggle notebook with GPU enabled.
2. Attached input dataset `voyari-tiny-day2-sft` containing `voyari_sft_step_008000.pt`.
3. Existing v2 training JSONL at `/kaggle/working/VoyariSML_Tiny/artifacts/data/sft_v2/mixed_train.jsonl`. If your Kaggle session was reset, restore it from your saved `Voyari_SFT_v2_Data_Backup.zip` dataset first.
4. A checkout of GitHub branch `training/stage1-stability` at `/kaggle/working/VoyariSML_Tiny_day3`.

## 1. Pull the committed trainer

```python
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
assert (repo / ".git").is_dir(), "Clone the training/stage1-stability branch first."
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
assert (repo / "scripts/train_sft_v2.py").is_file()
print("SFT v2 runner ready.")
```

## 2. Start the first 100-step experiment

```python
import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
checkpoint = Path(
    "/kaggle/input/datasets/prasadpattuvamveedu/"
    "voyari-tiny-day2-sft/voyari_sft_step_008000.pt"
)
data_dir = Path("/kaggle/working/VoyariSML_Tiny/artifacts/data/sft_v2")
tokenizer = repo / "artifacts/tokenizer/voyari_tokenizer_16k_v3.json"

for p in (checkpoint, data_dir / "mixed_train.jsonl", tokenizer):
    assert p.is_file(), f"Missing prerequisite: {p}"

command = [
    sys.executable, "-u", str(repo / "scripts/train_sft_v2.py"),
    "--base-checkpoint", str(checkpoint),
    "--data-dir", str(data_dir),
    "--files", "mixed_train.jsonl",
    "--tokenizer", str(tokenizer),
    "--run-name", "corrective_v1",
    "--max-steps", "100",
    "--session-steps", "100",
    "--max-length", "1024",
    "--grad-accum", "8",
    "--peak-lr", "0.000005",
    "--min-lr", "0.000001",
    "--warmup-steps", "10",
    "--val-fraction", "0.05",
    "--eval-samples", "32",
    "--eval-every", "25",
    "--save-every", "25",
    "--log-every", "10",
    "--keep-last", "4",
    "--resume", "auto",
]
subprocess.run(command, cwd=str(repo), check=True)
```

Checkpoints are saved under `/kaggle/working/VoyariSML_Tiny_day3/artifacts/checkpoints/sft_v2/corrective_v1/` and validation metrics under `logs/sft_v2/corrective_v1.jsonl`.

**Do not end the Kaggle session without preserving your new .pt checkpoints** (download them or save them in a private Kaggle Dataset). GitHub should contain scripts, configuration, and documentation, **not trained model binaries**.

## Important limitations

- The 960 mixed conversations are an exploratory corrective dataset, not a production-quality corpus. Their action-count summary is per assistant turn; one conversation may have multiple turns.
- Random validation samples from `mixed_train.jsonl` do not establish out-of-distribution behavior.
- Evaluate v2 against the **18 separate behavioral questions** already prepared in Kaggle; verify JSON validity, tool name, tool arguments, destination, budget, dates, duration, and explicit constraints. Correct action type alone is insufficient.
- Do not claim the model improved simply because training loss declined.

## 3. Evaluate the completed 100-step checkpoint

Run this after the runner successfully saved `voyari_sft_step_000100.pt` and your
`behavior_eval_independent.jsonl` is present in Kaggle.

```python
import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)

v2_data = Path("/kaggle/working/VoyariSML_Tiny/artifacts/data/sft_v2")
command = [
    sys.executable, "-u", str(repo / "scripts/eval_sft_v2.py"),
    "--checkpoint", str(
        repo / "artifacts/checkpoints/sft_v2/corrective_v1/voyari_sft_step_000100.pt"
    ),
    "--eval-file", str(v2_data / "behavior_eval_independent.jsonl"),
    "--system-dataset",
    "/kaggle/input/datasets/prasadpattuvamveedu/"
    "voyari-tiny-instruction-v1/01_voyari_v9_train.jsonl",
    "--output-dir", str(v2_data),
    "--max-new-tokens", "200",
]
subprocess.run(command, cwd=str(repo), check=True)
```

Read `sft_v2_independent_summary.json` and the per-question results.
For reference, the SFT v1 independent baseline was **15/18 valid JSON,
7/18 correct action types, and 4/6 correct tool names**. Its correct
tool-name count included outputs with wrong destinations, so check the
stricter tool-argument and state-field metrics before judging improvement.

**Keep evaluation files out of the training dataset.** Save the resulting
SFT v2 checkpoint and the latest evaluation JSONLs to a persistent private
Kaggle dataset or local backup before the session resets.


## 4. Comparison after the 100-step experiment

Both models were tested with the same 18 held-out behavioral questions:

| Metric | SFT v1 step 8000 | Corrective SFT v2 step 100 |
| --- | ---: | ---: |
| Valid JSON | 15/18 | 17/18 |
| Correct action type | 7/18 | 10/18 |
| Correct tool names | 4/6 | 4/6 |
| Exact tool arguments | 1/6 | 1/6 |
| Required state fields | 0/6 | 0/6 |
| Question-shaped clarifications | 0/6 | 4/6 |

The evaluator does **not** confirm whether questions are relevant or detect every invented extra field; for example, it separately reports `unexpected_state_fields`. This small diagnostic set does not justify extending training yet.

Run the data audit before expanding corrective SFT:

```python
import subprocess, sys
from pathlib import Path
repo = Path("/kaggle/working/VoyariSML_Tiny_day3")
data = Path("/kaggle/working/VoyariSML_Tiny/artifacts/data/sft_v2")
subprocess.run(
    ["git", "-C", str(repo), "pull", "--ff-only", "origin", "training/stage1-stability"],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-u", str(repo / "scripts/audit_sft_v2_data.py"),
        "--train", str(data / "mixed_train.jsonl"),
        "--eval", str(data / "behavior_eval_independent.jsonl"),
        "--report", str(data / "mixed_train_audit.json"),
    ],
    cwd=str(repo), check=True,
)
```

The audit flags potential invented dates, locations, duration, budget and traveller fields, but **these are only heuristic review flags**, not proof that an example is wrong. Inspect flagged conversations manually before editing data. Keep both model checkpoints backed up. The uploaded Custom RAG/Tool dataset needs schema conversion and quality filtering before it can join a later training run.
