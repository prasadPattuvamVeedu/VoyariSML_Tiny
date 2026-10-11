"""Day 5 synthetic SFT data: grounded origin/destination, budgets, tools, multi-turn.

NO model weights are loaded or updated here. Training data is generated only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from day5_schema import SYSTEM_TEXT, validate_action

CITIES = (
    "Kochi", "Jaipur", "Pune", "Thiruvananthapuram", "Delhi", "Mumbai", "Goa",
    "Munnar", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad",
    "Ooty", "Mysuru", "Kannur", "Kozhikode", "Thrissur", "Alappuzha",
    "Gangtok", "Hampi", "Sanchi", "Puri", "Nashik", "Manali", "Udaipur",
    "Jodhpur", "Varanasi", "Amritsar", "Shimla", "Dehradun", "Rishikesh",
    "Kanyakumari", "Rameswaram", "Palakkad", "Darjeeling", "Shillong",
    "Leh", "Mount Abu", "Madurai", "Bhopal", "Indore", "Lucknow", "Surat",
    "Vadodara", "Chandigarh", "Coimbatore", "Vijayawada", "Visakhapatnam",
    "Bhubaneswar", "Ranchi", "Puducherry", "Wayanad", "Kodaikanal", "Agra",
)
INTERESTS = ("museums", "food", "wildlife", "beaches", "heritage", "hiking",
             "temples", "photography", "history", "nature", "architecture")
DATES = (("2027-01-12", "2027-01-15"), ("2027-02-18", "2027-02-21"),
         ("2027-03-03", "2027-03-06"), ("2027-04-12", "2027-04-15"))


def norm(s):
    return " ".join(str(s).casefold().split())


def read_user_questions(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r.get("question"), str):
                yield norm(r["question"])
            for m in r.get("messages", []):
                if m.get("role") == "user":
                    yield norm(m.get("content", ""))


def load_system_dataset(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            for msg in json.loads(line).get("messages", []):
                if msg.get("role") == "system":
                    return msg["content"]
    raise ValueError("System dataset has no system instruction")


def amount(rng):
    return rng.choice(list(range(2000, 16001, 500)) + list(range(16500, 80001, 1500)))


def city(rng):
    return rng.choice(CITIES)


def two_cities(rng):
    return rng.sample(CITIES, 2)


def budget(rng):
    n = amount(rng)
    basis = rng.choices(["unknown", "total", "per_person"], [4, 4, 2])[0]
    surface = {"unknown": f"my budget is ₹{n:,}",
               "total": f"my total budget is ₹{n:,}",
               "per_person": f"I can spend ₹{n:,} per person"}[basis]
    slots = {"budget_inr": n}
    if basis != "unknown":
        slots["budget_basis"] = basis
    return surface, slots


def make(system, messages):
    return {"messages": [{"role": "system", "content": system}] + messages}


def reply(action):
    return {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))}


def single(system, q, action):
    return make(system, [{"role": "user", "content": q}, reply(action)])


def state_case(rng, system):
    variant = rng.randrange(9)
    dest = city(rng)
    orig = city(rng)
    while orig == dest:
        orig = city(rng)
    days = rng.randint(2, 11)
    btext, bs = budget(rng)
    if variant == 0:
        q = rng.choice((f"I want to visit {dest}.", f"My destination is {dest}.",
                        f"Please note that I'm going to {dest}.", f"Save {dest} as my destination."))
        s = {"destination": dest}
    elif variant == 1:
        q = rng.choice((f"I'm leaving from {orig}.", f"My starting city is {orig}.",
                        f"I will depart from {orig}.", f"Set my origin as {orig}."))
        s = {"origin": orig}
    elif variant == 2:
        q = rng.choice((f"I'm visiting {dest} for {days} days.",
                        f"My destination is {dest} and I have {days} days.",
                        f"Save a {days}-day trip to {dest}."))
        s = {"destination": dest, "duration_days": days}
    elif variant == 3:
        q = f"I'm planning a holiday in {dest}; {btext}."
        s = {"destination": dest, **bs}
    elif variant == 4:
        q = rng.choice((f"I want to go from {orig} to {dest}.",
                        f"Starting in {orig}, my destination is {dest}.",
                        f"I'm travelling from {orig} to {dest}."))
        s = {"origin": orig, "destination": dest}
    elif variant == 5:
        q = f"I'm travelling from {orig} to {dest} for {days} days."
        s = {"origin": orig, "destination": dest, "duration_days": days}
    elif variant == 6:
        q = f"I want to visit {dest} for {days} days; {btext}."
        s = {"destination": dest, "duration_days": days, **bs}
    elif variant == 7:
        interest = rng.choice(INTERESTS)
        q = f"My trip is to {dest} for {days} days and I'm interested in {interest}."
        s = {"destination": dest, "duration_days": days, "interests": [interest]}
    else:
        q = f"I am going from {orig} to {dest} for {days} days; {btext}."
        s = {"origin": orig, "destination": dest, "duration_days": days, **bs}
    return single(system, q, {"type": "state_update", "state": s})


def tool_case(rng, system):
    variant = rng.randrange(7)
    dest = city(rng)
    orig = city(rng)
    while orig == dest:
        orig = city(rng)
    days = rng.randint(2, 9)
    btext, bs = budget(rng)
    if variant == 0:
        when = rng.choice(("today", "tomorrow", "next Friday", "next weekend"))
        q = rng.choice((f"Check the weather in {dest} {when}.",
                        f"Please check the weather in {dest} {when}.",
                        f"Get the {when} weather forecast for {dest}."))
        a = {"type": "tool_call", "tool": "weather", "arguments": {"location": dest, "date_expression": when}}
    elif variant == 1:
        q = f"Find destination information about {dest}."
        a = {"type": "tool_call", "tool": "destination_knowledge", "arguments": {"query": dest}}
    elif variant in (2, 3):
        q = f"Search travel packages for {days} days in {dest}; {btext}."
        a = {"type": "tool_call", "tool": "search_travel_packages", "arguments": {
            "destination": dest, "duration_days": days, "max_budget_inr": bs["budget_inr"]}}
        if "budget_basis" in bs:
            a["arguments"]["budget_basis"] = bs["budget_basis"]
    elif variant == 4:
        arrive, leave = rng.choice(DATES)
        q = f"Search hotels in {dest} with check-in {arrive} and check-out {leave}."
        a = {"type": "tool_call", "tool": "search_hotels", "arguments": {
            "destination": dest, "check_in": arrive, "check_out": leave}}
    elif variant == 5:
        when = rng.choice(("tomorrow", "next Friday", "next weekend"))
        q = f"Search transport routes from {orig} to {dest} {when}."
        a = {"type": "tool_call", "tool": "search_transport", "arguments": {
            "origin": orig, "destination": dest, "date_expression": when}}
    else:
        interest = rng.choice(INTERESTS)
        q = f"Find {interest} activities in {dest}."
        a = {"type": "tool_call", "tool": "search_activities", "arguments": {
            "destination": dest, "interest": interest}}
    return single(system, q, a)


def clarification_case(rng, system):
    dest = city(rng)
    variant = rng.randrange(6)
    if variant == 0:
        q = rng.choice((f"How do I reach {dest}? I haven't specified where I'm leaving from.", f"Find a way to reach {dest}, but my origin is unknown.", f"I need travel routes to {dest}; I did not say where I start.", f"Please help me reach {dest}; my departure city is undecided."))
        ans = f"Which city are you travelling from to reach {dest}?"
    elif variant == 1:
        q = rng.choice((f"Find hotels in {dest}. I haven't chosen travel dates yet.", f"Check hotel availability in {dest}, but I have no check-in dates.", f"Show accommodation in {dest}, I don't know when I am going.", f"Find lodging in {dest}; dates are not selected."))
        ans = f"What are your check-in and check-out dates for {dest}?"
    elif variant == 2:
        q = rng.choice((f"I'd like to book a package to {dest}, but the departure dates are undecided.", f"Book a holiday package for {dest}; I have not picked dates.", f"Reserve a package to {dest}, travel dates are not known.", f"I want to book a tour to {dest}, but I do not know when."))
        ans = f"Which dates are you planning to travel to {dest}?"
    elif variant == 3:
        q = rng.choice((f"I am travelling to {dest} with a group. The number of people is undecided.", f"Our group trip is to {dest}; nobody has confirmed headcount.", f"We are going to {dest}, but group size is unknown.", f"Please record traveller count for our {dest} trip; it isn't decided."))
        ans = "How many travellers will be in your group?"
    elif variant == 4:
        q = rng.choice((f"My budget is ₹{amount(rng):,}, and I haven't selected a destination. Recommend a trip.", f"I have ₹{amount(rng):,} and need a destination; I did not say where I start.", f"Please suggest places under ₹{amount(rng):,}; I haven't chosen where to travel from."))
        ans = "Which city will you be travelling from?"
    else:
        q = rng.choice((f"I plan to travel from {dest} alone, but I also said I'll bring friends. Please save our group size.", f"My holiday starts in {dest}: I say solo and friends at the same time. How many travellers are we?", f"Starting from {dest}, I mentioned alone and also that friends join. Record traveller count.", f"From {dest}, I claimed solo travel but also a trip with friends. Please log group size."))
        ans = "Will you travel alone or with friends, and how many travellers are there?"
    return single(system, q, {"type": "message", "message": ans})


def multi_case(rng, system):
    variant = rng.randrange(3)
    dest = city(rng)
    days = rng.randint(2, 10)
    if variant == 0:  # A budget correction must replace only the given field.
        old, new = amount(rng), amount(rng)
        while old == new:
            new = amount(rng)
        user1 = f"I want to visit {dest} for {days} days and my budget is ₹{old:,}."
        a1 = {"type": "state_update", "state": {"destination": dest, "duration_days": days, "budget_inr": old}}
        user2 = f"Change my budget to ₹{new:,}."
        a2 = {"type": "state_update", "state": {"budget_inr": new}}
    elif variant == 1:  # Follow-up city disambiguated by preceding clarification.
        btext, bs = budget(rng)
        user1 = f"I want a trip for {days} days; {btext}, but I haven't selected my destination."
        a1 = {"type": "state_update", "state": {"duration_days": days, **bs}}
        user2 = f"My destination is {dest}."
        a2 = {"type": "state_update", "state": {"destination": dest}}
    else:  # Missing origin answered later.
        orig = city(rng)
        while orig == dest:
            orig = city(rng)
        user1 = f"How can I get to {dest}? My departure city is not stated."
        a1 = {"type": "message", "message": "Which city are you travelling from?"}
        user2 = f"I'm leaving from {orig}."
        a2 = {"type": "state_update", "state": {"origin": orig}}
    return make(system, [{"role": "user", "content": user1}, reply(a1),
                         {"role": "user", "content": user2}, reply(a2)])


def check_generated(record):
    last_user = ""
    for msg in record["messages"][1:]:
        if msg["role"] == "user":
            last_user = msg["content"]
        elif msg["role"] == "assistant":
            errors = validate_action(json.loads(msg["content"]), last_user)
            if errors:
                raise RuntimeError(f"Invalid generated label {errors}: {last_user} -> {msg['content']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--system-dataset", type=Path, help="Optional original V9 JSONL; uses exact known system text if absent")
    p.add_argument("--holdout", type=Path, action="append", default=[])
    p.add_argument("--existing-train", type=Path, action="append", default=[])
    p.add_argument("--count", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20261011)
    args = p.parse_args()
    if not 100 <= args.count <= 10000:
        p.error("count must be 100..10000")
    input_paths = [*args.holdout, *args.existing_train]
    if args.system_dataset:
        input_paths.append(args.system_dataset)
    for source in input_paths:
        if not source.is_file():
            raise FileNotFoundError(source)
    if len({args.output.resolve(), args.manifest.resolve(), *(x.resolve() for x in input_paths)}) != len(input_paths) + 2:
        raise ValueError("Output and manifest must not overwrite input files; inputs must be distinct")
    system = load_system_dataset(args.system_dataset) if args.system_dataset else SYSTEM_TEXT
    if not system.strip():
        raise ValueError("Empty system prompt")
    reserved = set()
    for source in input_paths:
        reserved.update(read_user_questions(source))
    rng = random.Random(args.seed)
    mix = Counter({"state": round(args.count * .525), "tool": round(args.count * .225),
                   "clarify": round(args.count * .15)})
    mix["multi"] = args.count - sum(mix.values())
    plan = [kind for kind, n in mix.items() for _ in range(n)]
    rng.shuffle(plan)
    build = {"state": state_case, "tool": tool_case, "clarify": clarification_case, "multi": multi_case}
    records, seen = [], set()
    for kind in plan:
        for attempt in range(2000):
            record = build[kind](rng, system)
            questions = [norm(m["content"]) for m in record["messages"] if m["role"] == "user"]
            if questions[0] not in seen and not any(q in reserved for q in questions):
                check_generated(record)
                seen.add(questions[0])
                records.append(record)
                break
        else:
            raise RuntimeError(f"Could not generate unique {kind} question after 2000 attempts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {"count": len(records), "mix": dict(mix), "seed": args.seed,
                "sha256": checksum, "holdout_overlap": 0,
                "source_files": [str(p) for p in input_paths],
                "note": "Synthetic; independent human label review and real blind evaluation are required before deployment."}
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DAY 5 DATA GENERATED", len(records), "conversations,", dict(mix))
    print("SHA256:", checksum)
    print("TRAIN:", args.output)
    print("MANIFEST:", args.manifest)


if __name__ == "__main__":
    main()
