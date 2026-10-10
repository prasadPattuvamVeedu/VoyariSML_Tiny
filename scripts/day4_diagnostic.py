"""Day 4 diagnostic evaluator for VoyariLM Tiny.

FROZEN development benchmark: do not train on eval/day4_diagnostic_v1.jsonl.
--check-only audits schema, duplicate prompts, and exact prompt overlap without GPU.
Without --check-only, evaluates SFT v1 or v2 using the existing greedy generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def args_parser():
    p = argparse.ArgumentParser(description="VoyariLM Tiny Day 4 diagnostic")
    p.add_argument("--eval-file", type=Path, default=ROOT / "eval/day4_diagnostic_v1.jsonl")
    p.add_argument("--training-file", type=Path, action="append", default=[])
    p.add_argument("--prior-eval-file", type=Path, action="append", default=[])
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--system-dataset", type=Path)
    p.add_argument("--tokenizer", type=Path,
                   default=ROOT / "artifacts/tokenizer/voyari_tokenizer_16k_v3.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/day4_evaluation")
    p.add_argument("--max-new-tokens", type=int, default=200)
    return p.parse_args()


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Invalid JSONL: {path}:{number}") from err


def norm(text):
    return " ".join(text.casefold().strip().split())


def questions_from_records(path):
    for record in read_jsonl(path):
        if isinstance(record.get("question"), str):
            yield record["question"]
        for message in record.get("messages", []):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                yield message["content"]


def benchmark_check(args):
    cases = list(read_jsonl(args.eval_file))
    if len(cases) != 30:
        raise ValueError(f"Expected 30 Day 4 cases, got {len(cases)}")
    counts = Counter()
    seen_ids = set()
    seen_questions = set()
    for case in cases:
        cid, category, question = case["id"], case["category"], case["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Blank diagnostic question")
        if cid in seen_ids or norm(question) in seen_questions:
            raise ValueError(f"Duplicate case ID or question: {cid}")
        seen_ids.add(cid)
        seen_questions.add(norm(question))
        counts[category] += 1
        expected = case["expected"]
        expected_type = {"state": "state_update", "tool": "tool_call",
                         "clarify": "message"}.get(category)
        if expected_type is None or expected.get("type") != expected_type:
            raise ValueError(f"{cid}: category/expected action mismatch")
        if category == "state" and not isinstance(expected.get("state"), dict):
            raise ValueError(f"{cid}: expected state object missing")
        if category == "tool" and (
            not isinstance(expected.get("arguments"), dict)
            or not isinstance(expected.get("tool"), str)
        ):
            raise ValueError(f"{cid}: expected tool/arguments missing")
        if category == "clarify" and (
            not isinstance(case.get("checks", {}).get("topic_any"), list)
            or not case["checks"]["topic_any"]
        ):
            raise ValueError(f"{cid}: missing clarification-topic check")
    if dict(counts) != {"state": 12, "tool": 8, "clarify": 10}:
        raise ValueError(f"Incorrect category distribution: {dict(counts)}")

    sources = [("train", p) for p in args.training_file] + [
        ("prior_eval", p) for p in args.prior_eval_file
    ]
    overlap = {}
    for kind, path in sources:
        other_questions = {norm(q) for q in questions_from_records(path)}
        shared = seen_questions.intersection(other_questions)
        if shared:
            raise RuntimeError(
                f"{len(shared)} exact Day 4 question overlaps in {kind}: {path} "
                f"(example: {sorted(shared)[0]})"
            )
        overlap[str(path)] = len(other_questions)

    digest = hashlib.sha256(args.eval_file.read_bytes()).hexdigest()
    print("DAY 4 BENCHMARK CHECK PASSED")
    print("Questions:", len(cases))
    print("Categories:", dict(counts))
    print("Checked source files:", len(sources))
    print("Exact overlap:", 0)
    print("Benchmark SHA256:", digest)
    return cases, digest


def canonical(value):
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, list):
        return [canonical(x) for x in value]
    if isinstance(value, dict):
        return {key: canonical(x) for key, x in value.items()}
    return value


def evaluate_answer(case, raw):
    try:
        predicted = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        predicted = None
    expected = case["expected"]
    result = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected": expected,
        "raw": raw,
        "predicted": predicted,
        "valid_json": isinstance(predicted, dict),
        "action_match": False,
        "exact_match": False,
        "required_fields_match": False,
        "unexpected_fields": [],
        "topic_match": None,
    }
    if not isinstance(predicted, dict):
        return result
    result["action_match"] = predicted.get("type") == expected["type"]
    if not result["action_match"]:
        return result

    if case["category"] == "state":
        got = predicted.get("state")
        gold = expected["state"]
        if isinstance(got, dict):
            result["required_fields_match"] = all(
                key in got and canonical(got[key]) == canonical(value)
                for key, value in gold.items()
            )
            result["unexpected_fields"] = sorted(set(got) - set(gold))
            result["exact_match"] = (
                result["required_fields_match"] and not result["unexpected_fields"]
            )
    elif case["category"] == "tool":
        got_args = predicted.get("arguments")
        gold_args = expected["arguments"]
        if isinstance(got_args, dict):
            result["required_fields_match"] = (
                predicted.get("tool") == expected["tool"]
                and all(
                    key in got_args and canonical(got_args[key]) == canonical(value)
                    for key, value in gold_args.items()
                )
            )
            result["unexpected_fields"] = sorted(set(got_args) - set(gold_args))
            result["exact_match"] = (
                result["required_fields_match"] and not result["unexpected_fields"]
            )
    else:
        message = predicted.get("message", "")
        if isinstance(message, str):
            terms = case["checks"]["topic_any"]
            result["topic_match"] = (
                "?" in message and any(term.casefold() in message.casefold() for term in terms)
            )
            result["exact_match"] = result["topic_match"]
    return result


def run_evaluation(args, cases, digest):
    if not args.checkpoint or not args.system_dataset:
        raise ValueError("--checkpoint and --system-dataset required for model evaluation")
    if args.max_new_tokens < 1 or args.max_new_tokens > 300:
        raise ValueError("Maximum output tokens must be between 1 and 300")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # Import model dependencies only when actual GPU inference is needed.
    import torch
    from tokenizers import Tokenizer
    from eval_sft_v2 import generate, system_from_dataset
    from src.model.voyari_lm import VoyariLM

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    stage, step = checkpoint.get("stage"), int(checkpoint.get("step", -1))
    if not ((stage == "sft" and step == 8000)
            or (stage == "sft_v2" and step == 100)):
        raise ValueError(f"Unsupported checkpoint: stage={stage} step={step}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    system_text = system_from_dataset(args.system_dataset)
    model = VoyariLM()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    results = []
    for index, case in enumerate(cases, 1):
        output = generate(
            model, tokenizer, device, system_text, case["question"], args.max_new_tokens
        )
        row = evaluate_answer(case, output)
        results.append(row)
        print(
            f"{index:02d}/30 [{row['category']:7}] "
            f"JSON={row['valid_json']} action={row['action_match']} "
            f"exact={row['exact_match']} extra={row['unexpected_fields']}",
            flush=True,
        )
    summary = {
        "checkpoint": str(args.checkpoint),
        "stage": stage,
        "step": step,
        "benchmark_sha256": digest,
        "device": str(device),
        "total": len(results),
        "valid_json": sum(r["valid_json"] for r in results),
        "action_match": sum(r["action_match"] for r in results),
        "exact_match_total": sum(r["exact_match"] for r in results),
        "categories": {},
        "caution": (
            "Development diagnostic; authored after known SFT v2 failures. "
            "State/tool exact scores penalize unsupported fields; clarification "
            "topic heuristic is not a substitute for human review."
        ),
    }
    for category in ("state", "tool", "clarify"):
        part = [r for r in results if r["category"] == category]
        summary["categories"][category] = {
            "total": len(part),
            "action_match": sum(r["action_match"] for r in part),
            "exact_or_topic_match": sum(r["exact_match"] for r in part),
            "unwanted_fields_cases": sum(bool(r["unexpected_fields"]) for r in part),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{stage}_step_{step:06d}"
    result_path = args.output_dir / f"day4_{label}_results.jsonl"
    summary_path = args.output_dir / f"day4_{label}_summary.json"
    with result_path.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nDAY 4 EVALUATION SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Result file:", result_path)
    print("Summary file:", summary_path)


def main():
    args = args_parser()
    cases, digest = benchmark_check(args)
    if not args.check_only:
        run_evaluation(args, cases, digest)


if __name__ == "__main__":
    main()
