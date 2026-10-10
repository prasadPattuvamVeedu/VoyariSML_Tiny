"""Evaluate corrective VoyariLM Tiny SFT v2 on fixed held-out travel prompts.

Reads existing checkpoints and JSONL evaluation cases; never trains or changes weights.
Outputs action, tool, argument, and state-field checks to a separate JSONL report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.voyari_lm import VoyariLM


def arguments():
    parser = argparse.ArgumentParser(description="Evaluate VoyariLM Tiny SFT v2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", type=Path,
        default=ROOT / "artifacts/tokenizer/voyari_tokenizer_16k_v3.json",
    )
    parser.add_argument("--system-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def system_from_dataset(path):
    records = read_jsonl(path)
    for record in records:
        for msg in record.get("messages", []):
            if msg.get("role") == "system":
                return str(msg["content"])
    raise ValueError(f"No system message found in {path}")


def canonical(value):
    """For a strict field comparison, allow whitespace/case changes only."""
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    return value


def score(expected, predicted):
    result = {
        "action_match": False,
        "tool_match": None,
        "arguments_match": None,
        "state_fields_match": None,
        "unexpected_state_fields": [],
        "clarifying_question": None,
    }
    if not isinstance(predicted, dict):
        return result
    kind = expected.get("type")
    result["action_match"] = predicted.get("type") == kind
    if kind == "tool_call":
        result["tool_match"] = (
            result["action_match"]
            and predicted.get("tool") == expected.get("tool")
        )
        expected_args = expected.get("arguments", {})
        got_args = predicted.get("arguments")
        result["arguments_match"] = (
            result["tool_match"]
            and isinstance(got_args, dict)
            and all(
                canonical(got_args.get(key)) == canonical(value)
                for key, value in expected_args.items()
            )
        )
    elif kind == "state_update":
        expected_state = expected.get("state", {})
        got_state = predicted.get("state")
        result["state_fields_match"] = (
            result["action_match"]
            and isinstance(got_state, dict)
            and all(
                canonical(got_state.get(key)) == canonical(value)
                for key, value in expected_state.items()
            )
        )
        if isinstance(got_state, dict):
            result["unexpected_state_fields"] = [
                key for key in got_state if key not in expected_state
            ]
    elif kind == "message":
        message = predicted.get("message", "")
        result["clarifying_question"] = (
            result["action_match"]
            and isinstance(message, str)
            and "?" in message
        )
    return result


@torch.inference_mode()
def generate(model, tokenizer, device, system_text, question, limit):
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("Tokenizer is missing <bos>/<eos> tokens")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    ids = [bos]
    for text in (
        "[system]\n", system_text, "\n",
        "[user]\n", question, "\n",
        "[assistant]\n",
    ):
        ids.extend(encode(text))
    if len(ids) + limit > 1024:
        raise ValueError("Prompt + output exceeds VoyariLM 1024-token context")

    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = []
    for _ in range(limit):
        logits = model(tokens)
        next_id = int(torch.argmax(logits[0, -1]).item())
        if next_id == eos:
            break
        output_ids.append(next_id)
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokenizer.decode(output_ids, skip_special_tokens=False)


def main():
    args = arguments()
    for path in (args.checkpoint, args.eval_file, args.tokenizer, args.system_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_new_tokens < 1 or args.max_new_tokens > 512:
        raise ValueError("--max-new-tokens must be between 1 and 512")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("stage") != "sft_v2":
        raise ValueError("Expected an SFT v2 checkpoint")
    if ckpt.get("run_config", {}).get("parent_sft_step") != 8000:
        raise ValueError("Expected parent SFT v1 step 8000")
    print("Evaluating SFT v2 optimizer step:", ckpt.get("step"), flush=True)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    system_text = system_from_dataset(args.system_dataset)
    cases = read_jsonl(args.eval_file)
    if not cases:
        raise ValueError("Evaluation file has no cases")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoyariLM()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    del ckpt
    model.to(device).eval()
    print("Device:", device, "| Questions:", len(cases), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "sft_v2_independent_results.jsonl"
    summary_file = args.output_dir / "sft_v2_independent_summary.json"

    results = []
    for i, case in enumerate(cases, 1):
        question = case["question"]
        expected = case["expected"]
        response = generate(
            model, tokenizer, device, system_text,
            question, args.max_new_tokens,
        )
        try:
            predicted = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            predicted = None
        valid_json = isinstance(predicted, dict)
        checks = score(expected, predicted)
        result = {
            "question": question,
            "expected": expected,
            "raw_response": response,
            "predicted": predicted,
            "valid_json": valid_json,
            **checks,
        }
        results.append(result)
        print(
            f"Test {i:02d}: JSON={valid_json} "
            f"action={checks['action_match']} "
            f"tool={checks['tool_match']} "
            f"args={checks['arguments_match']} "
            f"state={checks['state_fields_match']}",
            flush=True,
        )
        print("  Output:", response[:300].replace("\n", " "), flush=True)

    def count(field, subset=None):
        selected = (
            results if subset is None
            else [r for r in results if r["expected"]["type"] == subset]
        )
        return sum(r.get(field) is True for r in selected), len(selected)

    summary = {
        "checkpoint": str(args.checkpoint),
        "eval_file": str(args.eval_file),
        "total": len(results),
        "valid_json": count("valid_json"),
        "action_match": count("action_match"),
        "tool_name_match": count("tool_match", "tool_call"),
        "tool_arguments_match": count("arguments_match", "tool_call"),
        "state_fields_match": count("state_fields_match", "state_update"),
        "clarifying_question": count("clarifying_question", "message"),
        "note": (
            "Exact structured-field checks; clarification relevance, "
            "hallucination, and safety require manual review. "
            "Do not treat small test-set scores as population accuracy."
        ),
    }

    with output_file.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nSFT V2 INDEPENDENT EVALUATION")
    for key, value in summary.items():
        if isinstance(value, tuple):
            print(f"{key}: {value[0]}/{value[1]}")
        elif isinstance(value, list):
            print(f"{key}: {value[0]}/{value[1]}")
    print("Results:", output_file)
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
