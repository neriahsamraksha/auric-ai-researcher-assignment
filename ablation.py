import argparse
import hashlib
import json
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import main
import score

RESULTS = main.ROOT / "results"
ARMS = ("E", "N", "D")
BUDGETS = (("1x", 1), ("3x", 3), ("10x", 10))
CASES = ("unique", "ambiguous", "inconsistent")
SELECTABLE = ("single", "fixed_vote", "agreement", "trigger", "answer_vote")
SELECTION_RULE = ("a policy using more calls replaces a cheaper one only if its dev-half macro score "
                  "is higher in every recorded run")

DIRECT_PROMPT = """You solve staff rota puzzles from shift notes. Every person works exactly one block, every block has exactly one person, and each station has exactly one holder chosen from the people the first line says are on a station. Hedged statements count as fully true. Remarks about wishes, past arrangements, hypotheticals or unsettled discussions are not constraints.

Decide which case applies and answer with one JSON object and nothing else:
- exactly one assignment fits: {"case": "unique", "assignment": {"<name>": {"block": "<time>", "station": "<station>"}, ...}}
- two to four assignments fit: {"case": "ambiguous", "assignments": [<assignment>, ...]} listing every one
- no assignment fits: {"case": "inconsistent", "conflicts": ["<statement>", ...]} quoting, in full and verbatim, a smallest set of statements that cannot all hold.
Name every person on the rota and give "station" only for station holders. Copy names, times, stations and statements exactly."""

CONDITIONS = (
    ("full", "Full system (shipped policy)", "E", "selected", "smallest", True),
    ("single", "Multi-call off: one call", "E", "single", "smallest", True),
    ("fixed_vote", "Line vote over all calls", "E", "fixed_vote", "smallest", True),
    ("answer_vote", "Answer vote over all calls", "E", "answer_vote", "smallest", True),
    ("agreement", "Stop when two answers agree", "E", "agreement", "smallest", True),
    ("trigger", "Stop when extraction is clean", "E", "trigger", "smallest", True),
    ("core_deletion", "Conflict set by deletion order", "E", "selected", "deletion", True),
    ("core_all", "Conflict minimisation off", "E", "selected", "all", True),
    ("strict_parse", "Parser normalisation off", "E", "selected", "smallest", False),
    ("no_guidance", "Distractor guidance removed (arm N)", "N", "selected", "smallest", True),
    ("direct", "Solver removed: direct answer, answer vote (arm D)", "D", None, None, None),
)


def prompt_hash_for(arm):
    if arm == "D":
        return hashlib.sha256(DIRECT_PROMPT.encode("utf-8")).hexdigest()[:12]
    return main.prompt_hash(guidance=arm == "E")


def messages_for(arm, raw):
    if arm == "D":
        return [{"role": "system", "content": DIRECT_PROMPT}, {"role": "user", "content": raw["text"]}]
    return main.build_messages(main.parse_item(raw), guidance=arm == "E")


def load_samples(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records.setdefault(record["item"], []).append(record)
    for item_records in records.values():
        item_records.sort(key=lambda record: record["sample"])
    return records


def check_hash(path, records, expected):
    if {r["prompt_hash"] for rs in records.values() for r in rs} - {expected}:
        raise main.ConfigError(f"{path.name} was recorded with a different prompt version; "
                               "move it out of results/ or use a new run number")


def load_runs(arm):
    runs = {}
    for path in sorted(RESULTS.glob(f"samples_{arm}_r*.jsonl")):
        records = load_samples(path)
        check_hash(path, records, prompt_hash_for(arm))
        runs[int(path.stem.rsplit("_r", 1)[1])] = records
    return runs


def collect(args):
    if not 1 <= args.samples <= 10:
        raise main.ConfigError("--samples must be between 1 and 10")
    items = main.load_items(args.items)
    client = main.make_client()
    workers = main.worker_count()
    expected = prompt_hash_for(args.arm)
    path = RESULTS / f"samples_{args.arm}_r{args.run}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_samples(path)
    check_hash(path, existing, expected)
    done = {item for item, records in existing.items() if len(records) >= args.samples}
    todo = [raw for raw in items if raw["id"] not in done]
    lock = threading.Lock()

    def work(raw):
        messages = messages_for(args.arm, raw)
        budget = main.CallBudget(args.samples)
        lines, failures = [], 0
        for index in range(args.samples):
            reply = main.request_completion(client, raw["id"], messages, budget)
            failures += reply["error"] is not None
            lines.append(json.dumps({
                "arm": args.arm, "run": args.run, "item": raw["id"], "sample": index,
                "ok": reply["error"] is None, "output": reply["text"], "error_type": reply["error"],
                "finish_reason": reply["finish_reason"], "latency_s": reply["latency_s"],
                "model_returned": reply["model"], "prompt_hash": expected}, ensure_ascii=False))
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return budget.used, failures

    with ThreadPoolExecutor(max_workers=workers) as pool:
        totals = list(pool.map(work, todo))
    calls = sum(used for used, _ in totals)
    failures = sum(failed for _, failed in totals)
    print(f"arm {args.arm} run {args.run}: {len(todo)} items collected ({len(done)} already present), "
          f"{calls} model calls, {failures} failed calls -> {path}", file=sys.stderr)


def parse_direct(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("case") in CASES:
            return value
    return None


def simulate(contexts, records, policy, cap, core_rule, normalise):
    answers, calls = {}, 0
    for item_id, ctx in contexts.items():
        item_records = records.get(item_id, [])
        limit = min(cap, len(item_records))
        if not limit:
            continue

        def draw(index, item_records=item_records):
            record = item_records[index]
            return record["output"] if record["ok"] else None

        try:
            result = main.run_policy(policy, limit, draw, ctx, normalise, core_rule)
        except Exception:
            calls += limit
            continue
        calls += result["calls"]
        if result["outcome"] == "answered":
            answers[item_id] = result["answer"]
    return answers, calls / len(contexts)


def simulate_direct(records, ids, cap):
    answers, calls = {}, 0
    for item_id in ids:
        item_records = records.get(item_id, [])[:cap]
        calls += len(item_records)
        parsed = [parse_direct(r["output"]) for r in item_records if r["ok"]]
        parsed = [answer for answer in parsed if answer is not None]
        if parsed:
            answers[item_id] = main.majority(parsed)
    return answers, calls / len(ids)


def scored(key, items, answers, ids):
    subset = set(ids)
    result = score.score({i: key[i] for i in ids}, {i: a for i, a in answers.items() if i in subset},
                         [raw for raw in items if raw["id"] in subset])
    result.pop("per_item", None)
    return result


def cell(row):
    if not row["runs"]:
        return "not run"
    values = [r["all"]["macro_exact_match"] for r in row["runs"]]
    calls = statistics.mean(r["mean_calls"] for r in row["runs"])
    return f"{statistics.mean(values):.3f} [{', '.join(f'{v:.3f}' for v in values)}], {calls:.1f} calls"


def evaluate(args):
    items = main.load_items(args.items)
    try:
        with open(args.key, encoding="utf-8") as handle:
            key = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise main.ConfigError(f"cannot read key file {args.key}: {exc}") from None
    ids = [raw["id"] for raw in items]
    halves = {"all": ids, "dev": ids[: len(ids) // 2], "rest": ids[len(ids) // 2:]}
    contexts = {raw["id"]: main.prepare(main.parse_item(raw)) for raw in items}
    runs = {arm: load_runs(arm) for arm in ARMS}
    if not runs["E"]:
        raise main.ConfigError("no recorded runs for arm E; collect them first")
    cache = {}

    def measure(arm, run, policy, cap, core_rule, normalise):
        token = (arm, run, policy, cap, core_rule, normalise)
        if token not in cache:
            records = runs[arm][run]
            if arm == "D":
                answers, calls = simulate_direct(records, ids, cap)
            else:
                answers, calls = simulate(contexts, records, policy, cap, core_rule, normalise)
            cache[token] = {**{name: scored(key, items, answers, subset) for name, subset in halves.items()},
                            "mean_calls": round(calls, 3), "answered": len(answers)}
        return cache[token]

    selected, policy_rows = {"1x": "single"}, []
    for budget, cap in BUDGETS[1:]:
        stats = []
        for policy in SELECTABLE:
            per_run = [measure("E", run, policy, cap, "smallest", True) for run in sorted(runs["E"])]
            dev = [m["dev"]["macro_exact_match"] for m in per_run]
            calls = statistics.mean(m["mean_calls"] for m in per_run)
            stats.append((calls, SELECTABLE.index(policy), policy, dev))
            policy_rows.append({"budget": budget, "policy": policy, "mean_calls": round(calls, 3),
                                "dev_macro_per_run": dev,
                                "all_macro_mean": round(statistics.mean(m["all"]["macro_exact_match"] for m in per_run), 4)})
        stats.sort()
        choice = stats[0]
        for candidate in stats[1:]:
            if all(c > s for c, s in zip(candidate[3], choice[3])):
                choice = candidate
        selected[budget] = choice[2]

    rows = []
    for name, label, arm, policy, core_rule, normalise in CONDITIONS:
        for budget, cap in BUDGETS:
            chosen = main.SHIPPED_POLICY[budget] if policy == "selected" else policy
            row = {"condition": name, "label": label, "arm": arm, "budget": budget, "policy": chosen,
                   "core_rule": core_rule, "normalise": normalise, "runs": []}
            for run in sorted(runs[arm]):
                row["runs"].append({"run": run, **measure(arm, run, chosen, cap, core_rule, normalise)})
            rows.append(row)

    counts = ", ".join(f"arm {arm}: {len(runs[arm])} run(s)" for arm in ARMS)
    lines = ["# Ablation results", "",
             f"Recorded samples: {counts}. Prompt versions: "
             + ", ".join(f"{arm}={prompt_hash_for(arm)}" for arm in ARMS) + ".",
             "Budget k replays the first k recorded samples of each item. Each cell: mean macro exact match on all "
             "items [value per run], mean model calls per item.",
             f"Shipped policy: 3x = {main.SHIPPED_POLICY['3x']}, 10x = {main.SHIPPED_POLICY['10x']}. Rule selects: 3x = {selected['3x']}, 10x = {selected['10x']} ({SELECTION_RULE}; dev half = first "
             f"{len(halves['dev'])} items of items.json).", "",
             "| Condition | 1x | 3x | 10x |", "|---|---|---|---|"]
    for name, label, *_ in CONDITIONS:
        cells = [cell(row) for row in rows if row["condition"] == name]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| Budget | Policy | Mean calls | Dev-half macro per run | All-items macro (mean) |",
              "|---|---|---|---|---|"]
    for p in policy_rows:
        lines.append(f"| {p['budget']} | {p['policy']} | {p['mean_calls']:.2f} | "
                     f"{', '.join(f'{v:.3f}' for v in p['dev_macro_per_run'])} | {p['all_macro_mean']:.3f} |")
    table = "\n".join(lines) + "\n"
    main.write_json_atomic(RESULTS / "ablation.json", {
        "shipped": main.SHIPPED_POLICY, "selected": selected, "selection_rule": SELECTION_RULE, "halves": halves,
        "runs": {arm: sorted(runs[arm]) for arm in ARMS},
        "prompt_hashes": {arm: prompt_hash_for(arm) for arm in ARMS},
        "policy_selection": policy_rows, "rows": rows})
    (RESULTS / "ablation_table.md").write_text(table, encoding="utf-8")
    print(table)


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Record model samples and evaluate the ablation matrix.")
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect", help="record model samples (makes API calls)")
    collect_parser.add_argument("--arm", required=True, choices=ARMS)
    collect_parser.add_argument("--run", required=True, type=int)
    collect_parser.add_argument("--samples", type=int, default=10)
    collect_parser.add_argument("--items", default="items.json")
    evaluate_parser = commands.add_parser("evaluate", help="score all conditions from recorded samples (no API calls)")
    evaluate_parser.add_argument("--key", required=True)
    evaluate_parser.add_argument("--items", default="items.json")
    args = parser.parse_args(argv)
    try:
        collect(args) if args.command == "collect" else evaluate(args)
    except main.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())