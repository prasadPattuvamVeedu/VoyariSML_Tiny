"""Generate varied, grounded Day 4 state-extraction SFT data.

Produces assistant-only SFT labels with exactly the slots supported by user text.
Creates synthetic *training* conversations, NOT model answers or evaluation cases.
Always keep eval/day4_diagnostic_v1.jsonl out of training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

# Deliberately distinct from the Day 4 diagnostic state cities.
ORIGINS = (
    "Kochi", "Delhi", "Mumbai", "Kolkata", "Pune", "Chennai",
    "Bengaluru", "Jaipur", "Hyderabad", "Ahmedabad", "Lucknow",
    "Nagpur", "Indore", "Chandigarh", "Patna", "Jabalpur",
    "Kanpur", "Vijayawada", "Dehradun", "Panaji", "Ranchi",
    "Jamshedpur", "Thrissur", "Kannur", "Kozhikode",
)
INTERESTS = (
    "photography", "nature", "food", "heritage", "hiking",
    "wildlife", "temples", "museums", "street food", "architecture",
    "mountains", "beaches", "local experiences", "history",
    "birdwatching", "art", "waterfalls", "forts",
)
DAY_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven",
}
INTRO = (
    "Please save the travel details I have so far.",
    "Update my trip profile using only the information here.",
    "Record these trip preferences accurately.",
    "Keep my stated travel details for later planning.",
    "I want to store my trip information.",
    "Add these known details to my trip state.",
    "Please note these constraints without guessing other values.",
    "I am organizing a holiday; save what I mention.",
    "Capture the trip facts below.",
    "Write down my current travel requirements.",
    "Can you remember the following trip information?",
)
ORIGIN_TEXT = (
    "I'm leaving from {origin}.",
    "My starting city is {origin}.",
    "The journey starts in {origin}.",
    "I'll depart from {origin}.",
    "I begin the trip in {origin}.",
)
DURATION_TEXT = (
    "I have {days} days.",
    "The trip lasts {days} days.",
    "I'll travel for {days} days.",
    "My holiday length is {days} days.",
    "I plan to be away for {days} days.",
    "The duration is {days} days.",
)
GROUP_TEXT = {
    "solo": ("I'll be travelling solo.", "This is a solo trip.",
             "I'll travel alone.", "It's just me travelling."),
    "couple": ("I am travelling with my partner.", "I'll go with my spouse.",
               "I will travel with my partner."),
    "family_parents": ("I'll be travelling with my parents.",
                       "I'm going with my parents."),
    "friends": ("I'm travelling with friends.", "I'll be going with my friends."),
    "family": ("I'm travelling with family.", "I'll go with my family."),
}
BUDGET_TEXT = {
    "total": ("My total budget is ₹{budget}.", "I can spend ₹{budget} altogether.",
              "The full trip budget is ₹{budget}."),
    "per_person": ("The budget is ₹{budget} per person.",
                   "Each traveller can spend ₹{budget}.",
                   "My per-person budget is ₹{budget}."),
}
INTEREST_TEXT = (
    "I'm interested in {interest}.",
    "I prefer {interest}.",
    "I especially enjoy {interest}.",
    "My main travel interest is {interest}.",
)
NO_BUDGET = (
    "The budget isn't decided yet.",
    "I haven't selected a budget.",
    "I don't know my spending limit yet.",
)
NO_GROUP = (
    "The number of travellers isn't decided.",
    "The group size is still unknown.",
    "I haven't settled the party size yet.",
)
NO_DATE = (
    "I have not selected travel dates.",
    "The calendar dates are still undecided.",
    "I don't know the departure date yet.",
)
QUESTION_VARIATIONS = (
    "I need hotel availability in {city}, but I haven't picked the dates.",
    "Can you find a stay in {city}? My check-in and check-out are undecided.",
    "Look for a hotel in {city}. I cannot give travel dates yet.",
    "Please search rooms in {city}, although my travel dates are unknown.",
    "Show accommodation choices for {city}; I don't know the dates yet.",
    "I need a hotel in {city} but have not selected check-in or check-out dates.",
)
GROUP_CONFLICT = (
    "I say this is a solo trip from {city}, but our group plans to travel together. Please save the group size.",
    "I am going alone from {city}, but I also said all of us are travelling. Record the traveller count.",
    "I mentioned travelling solo from {city} and also with my friends. Can you set the group size?",
    "I will start from {city} alone, but I also mentioned our group. Please set the traveller count.",
    "Travelling from {city}: I said solo and also that all of us are going. How many people should be recorded?",
)
TRANSPORT_MISSING_ORIGIN = (
    "Find transport to {city} tomorrow, but my departure city is unknown.",
    "Look up buses to {city} next weekend. I haven't said where I'm starting.",
    "Check travel routes to {city} next Friday; my starting point is not set.",
    "Search transport options to {city} tomorrow. I did not provide my departure city.",
)
TOOL_WEATHER_DATE = ("tomorrow", "today", "next Friday", "next weekend")
TOOL_WEATHER_TEXT = (
    "Please check the weather in {city} {when}.",
    "Retrieve the {when} weather forecast for {city}.",
    "What's the weather expected to be in {city} {when}? Check it.",
)
KNOWLEDGE_TEXT = (
    "Find destination information about {city}.",
    "Search travel knowledge for {city}.",
    "Look up a destination overview for {city}.",
)


def normalize(text):
    return " ".join(str(text).casefold().strip().split())


def first_user(record):
    if isinstance(record.get("question"), str):
        return normalize(record["question"])
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return normalize(msg.get("content", ""))
    return None


def load_system(path):
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            record = json.loads(raw)
            for msg in record.get("messages", []):
                if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    return msg["content"]
    raise ValueError("No system instruction in --system-dataset")


def read_questions(path):
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            if raw.strip():
                value = first_user(json.loads(raw))
                if value:
                    yield value


def choose(rng, seq):
    return rng.choice(seq)


def state_case(rng):
    origin = choose(rng, ORIGINS)
    days = rng.randint(2, 11)
    duration = DAY_WORDS[days] if days in DAY_WORDS and rng.random() < 0.4 else str(days)
    state = {"origin": origin, "duration_days": days}
    parts = [
        choose(rng, ORIGIN_TEXT).format(origin=origin),
        choose(rng, DURATION_TEXT).format(days=duration),
    ]
    if rng.random() < 0.65:
        group = choose(rng, tuple(GROUP_TEXT))
        group_state = {"solo": ("solo", 1), "couple": ("couple", 2),
                       "family_parents": ("family_parents", 3),
                       "friends": ("friends", None), "family": ("family", None)}
        name, count = group_state[group]
        state["traveller_group"] = name
        if count is not None:
            state["traveller_count"] = count
        parts.append(choose(rng, GROUP_TEXT[group]))
    elif rng.random() < 0.5:
        parts.append(choose(rng, NO_GROUP))
    if rng.random() < 0.75:
        amount = rng.choice(range(6500, 76001, 500))
        basis = choose(rng, ("total", "per_person"))
        state["budget_inr"] = amount
        state["budget_basis"] = basis
        parts.append(choose(rng, BUDGET_TEXT[basis]).format(
            budget=f"{amount:,}"
        ))
    elif rng.random() < 0.7:
        parts.append(choose(rng, NO_BUDGET))
    if rng.random() < 0.65:
        interest = choose(rng, INTERESTS)
        state["interests"] = [interest]
        parts.append(choose(rng, INTEREST_TEXT).format(interest=interest))
    if rng.random() < 0.35:
        parts.append(choose(rng, NO_DATE))
    rng.shuffle(parts)
    question = choose(rng, INTRO) + " " + " ".join(parts)
    return question, {"type": "state_update", "state": state}, "state"


def clarification_case(rng):
    choice = rng.randrange(3)
    city = choose(rng, ORIGINS)
    if choice == 0:
        return (choose(rng, QUESTION_VARIATIONS).format(city=city),
                {"type": "message", "message": (
                    f"What check-in and check-out dates are you considering for {city}?"
                )}, "clarify")
    if choice == 1:
        return (choose(rng, GROUP_CONFLICT).format(city=city),
                {"type": "message", "message": (
                    "You mentioned conflicting traveller details. "
                    "Will you be travelling alone or with a group?"
                )}, "clarify")
    return (choose(rng, TRANSPORT_MISSING_ORIGIN).format(city=city),
            {"type": "message", "message": (
                f"Which city will you be departing from to travel to {city}?"
            )}, "clarify")


def tool_case(rng):
    city = choose(rng, ORIGINS)
    if rng.random() < 0.7:
        when = choose(rng, TOOL_WEATHER_DATE)
        return (choose(rng, TOOL_WEATHER_TEXT).format(city=city, when=when),
                {"type": "tool_call", "tool": "weather", "arguments": {
                    "location": city, "date_expression": when,
                }}, "tool")
    return (choose(rng, KNOWLEDGE_TEXT).format(city=city),
            {"type": "tool_call", "tool": "destination_knowledge",
             "arguments": {"query": city}}, "tool")


def make_record(system, question, answer):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": json.dumps(
            answer, ensure_ascii=False, separators=(",", ":")
        )},
    ]}


def main():
    parser = argparse.ArgumentParser(description="Generate grounded Day 4 SFT data")
    parser.add_argument("--system-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, action="append", default=[])
    parser.add_argument("--existing-train", type=Path, action="append", default=[])
    parser.add_argument("--state", type=int, default=1200)
    parser.add_argument("--clarify", type=int, default=240)
    parser.add_argument("--tool", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    paths = [args.system_dataset, *args.holdout, *args.existing_train]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    # The system dataset can legitimately be listed again for train-overlap checks.
    # Only output/manifest must be distinct from each other and every input.
    input_paths = {p.resolve() for p in paths}
    output_path, manifest_path = args.output.resolve(), args.manifest.resolve()
    if (output_path == manifest_path
            or output_path in input_paths
            or manifest_path in input_paths):
        raise ValueError("Output and manifest must not overwrite any input files")
    if any(x < 0 for x in (args.state, args.clarify, args.tool)):
        raise ValueError("Example counts must be nonnegative")
    requested = args.state + args.clarify + args.tool
    if requested == 0 or requested > 10000:
        raise ValueError("Total examples must be between 1 and 10000")

    system = load_system(args.system_dataset)
    reserved = set()
    source_question_counts = {}
    for path in args.holdout + args.existing_train:
        values = set(read_questions(path))
        reserved.update(values)
        source_question_counts[str(path)] = len(values)

    randomizer = random.Random(args.seed)
    examples = []
    kinds = Counter()
    used = set()
    kinds_plan = ["state"] * args.state + ["clarify"] * args.clarify + ["tool"] * args.tool
    randomizer.shuffle(kinds_plan)
    generators = {"state": state_case, "clarify": clarification_case, "tool": tool_case}
    for kind in kinds_plan:
        for _attempt in range(500):
            question, answer, actual_kind = generators[kind](randomizer)
            canonical = normalize(question)
            if canonical in reserved or canonical in used:
                continue
            if actual_kind != kind:
                raise AssertionError("Generation kind mismatch")
            used.add(canonical)
            examples.append(make_record(system, question, answer))
            kinds[kind] += 1
            break
        else:
            raise RuntimeError(f"Unable to create unique {kind} case")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    content_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "generated_file": str(args.output),
        "generator_seed": args.seed,
        "sha256": content_sha,
        "cases": len(examples),
        "categories": dict(kinds),
        "reserved_sources": source_question_counts,
        "exact_prompt_overlap": 0,
        "training_city_pool": list(ORIGINS),
        "note": (
            "Synthetic templates provide slot supervision but are not independent "
            "evidence of model generalization. Do not include the Day 4 benchmark "
            "in training. Audit for semantic quality before any fine-tuning."
        ),
    }
    args.manifest.write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")
    print("DAY 4 TARGETED DATA GENERATED")
    print("Training examples:", len(examples))
    print("Categories:", dict(kinds))
    print("Exact overlaps with prior/held-out prompts:", 0)
    print("Generator SHA256:", content_sha)
    print("Training file:", args.output)
    print("Manifest:", args.manifest)


if __name__ == "__main__":
    main()
