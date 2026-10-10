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
