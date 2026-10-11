"""Compare VoyariLM checkpoint against Day 5 development questions.

Read-only inference. Uses the same greedy generator and prompt format as the
existing scripts/eval_sft_v2.py. This dev set is NOT a blind final test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def score_case(case, predicted):
    expected = case["expected"]
    category = case["category"]
    outcome = {"valid_json": isinstance(predicted, dict), "action_match": False,
               "exact_or_topic_match": False, "missing": [], "extra": [], "incorrect": []}
    if not isinstance(predicted, dict):
        return outcome
    if predicted.get("type") != expected["type"]:
        return outcome
    outcome["action_match"] = True
    if category == "clarify":
        message = predicted.get("message", "")
        outcome["exact_or_topic_match"] = isinstance(message, str) and "?" in message and any(
            t.casefold() in message.casefold() for t in case["checks"]["topic_any"])
        return outcome
    if category == "tool" and predicted.get("tool") != expected.get("tool"):
        return outcome
    key = "arguments" if category == "tool" else "state"
    actual = predicted.get(key)
    gold = expected[key]
    if not isinstance(actual, dict):
        outcome["missing"] = sorted(gold)
        return outcome
    outcome["missing"] = sorted(set(gold) - set(actual))
    outcome["extra"] = sorted(set(actual) - set(gold))
    outcome["incorrect"] = sorted(k for k in gold.keys() & actual.keys() if
                                  str(gold[k]).casefold() != str(actual[k]).casefold())
    outcome["exact_or_topic_match"] = not any(outcome[k] for k in ("missing", "extra", "incorrect"))
    return outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/voyari_tokenizer_16k_v3.json"))
    p.add_argument("--eval-file", type=Path, default=Path("eval/day5_dev_v1.jsonl"))
    p.add_argument("--system-dataset", type=Path, help="Optional original V9 system dataset")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/sft_day5"))
    args = p.parse_args()
    if not 1 <= args.max_new_tokens <= 300:
        p.error("max_new_tokens must be in 1..300")
    for path in (args.checkpoint, args.tokenizer, args.eval_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    from tokenizers import Tokenizer
    from src.model.voyari_lm import VoyariLM
    from eval_sft_v2 import generate
    from generate_day5_training import load_system_dataset
    from day5_schema import SYSTEM_TEXT

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") not in {"sft_day4", "sft_day5"}:
        raise ValueError("Use a Day 4 or Day 5 checkpoint, not pretraining")
    expected_sha = checkpoint.get("run_config", {}).get("tokenizer_sha256")
    actual_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    if expected_sha and expected_sha.lower() != actual_sha.lower():
        raise ValueError("Exact Kaggle tokenizer hash mismatch; do not disable this check")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = VoyariLM()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    stage = checkpoint["stage"]
    step = int(checkpoint.get("step", -1))
    del checkpoint
    system = load_system_dataset(args.system_dataset) if args.system_dataset else SYSTEM_TEXT
    cases = [json.loads(line) for line in args.eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        raw = generate(model, tokenizer, device, system, case["question"], args.max_new_tokens)
        try:
            predicted = json.loads(raw)
        except json.JSONDecodeError:
            predicted = None
        result = score_case(case, predicted)
        rows.append({**case, "raw_response": raw, "predicted": predicted, **result})
        print(f"{i:02d}/{len(cases)} {case['category']:7s} action={result['action_match']} exact_or_topic={result['exact_or_topic_match']}", flush=True)
    summary = {"checkpoint": str(args.checkpoint), "stage": stage, "step": step,
               "total": len(rows), "valid_json": sum(x["valid_json"] for x in rows),
               "action_match": sum(x["action_match"] for x in rows),
               "exact_or_topic_match": sum(x["exact_or_topic_match"] for x in rows),
               "categories": {}}
    for category in ("state", "tool", "clarify"):
        group = [x for x in rows if x["category"] == category]
        summary["categories"][category] = {
            "total": len(group), "action_match": sum(x["action_match"] for x in group),
            "exact_or_topic_match": sum(x["exact_or_topic_match"] for x in group)}
    summary["note"] = "Repeated development evaluation can overfit. Use a separate new unseen blind test for acceptance."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage}_step_{step:06d}_day5_dev"
    result_file = args.output_dir / f"{prefix}_results.jsonl"
    summary_file = args.output_dir / f"{prefix}_summary.json"
    result_file.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DAY 5 DEV EVALUATION\n", json.dumps(summary, indent=2))
    print("Results:", result_file, "\nSummary:", summary_file)


if __name__ == "__main__":
    main()
