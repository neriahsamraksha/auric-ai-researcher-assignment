import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_ID = "ibm-granite/granite-4.2-8b"
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 2000
TIMEOUT_SECONDS = 120
BUDGETS = {"1x": 1, "3x": 3, "10x": 10}
POLICIES = ("single", "fixed_vote", "agreement", "trigger", "answer_vote")
SHIPPED_POLICY = {"1x": "single", "3x": "trigger", "10x": "trigger"}  # selected by the pre-registered rule in ablation.py
MAX_PLAUSIBLE_SOLUTIONS = 4
ROOT = Path(__file__).resolve().parent


class ConfigError(Exception):
    pass


class HeaderError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


# Configuration

def load_env_file(path):
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def make_client():
    load_env_file(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if not api_key or not base_url:
        raise ConfigError("OPENROUTER_API_KEY and OPENROUTER_BASE_URL must be set in the environment or .env")
    model = os.environ.get("MODEL", MODEL_ID)
    if model != MODEL_ID:
        raise ConfigError(f"MODEL must be {MODEL_ID}")
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=TIMEOUT_SECONDS)


def worker_count():
    value = os.environ.get("AURIC_WORKERS", "4")
    if not value.isdigit() or int(value) < 1:
        raise ConfigError("AURIC_WORKERS must be a positive integer")
    return int(value)


# Header parsing

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def split_names(text):
    return [part.strip() for part in re.split(r",|\band\b", text) if part.strip()]


def header_field(header, pattern, label):
    match = re.search(pattern, header)
    if not match:
        raise HeaderError(f"header has no {label}")
    return match.group(1)


def parse_item(raw):
    item_id, text = raw.get("id"), raw.get("text")
    if not isinstance(item_id, str) or not isinstance(text, str):
        raise HeaderError("item needs string 'id' and 'text'")
    header, _, body = text.partition("\n")
    people = split_names(header_field(header, r"staff on the rota:\s*([^.]+)\.", "staff list"))
    stations = split_names(header_field(header, r"one person on each:\s*([^.]+)\.", "station list"))
    holders = split_names(header_field(header, r"people on a station are\s*([^;.]+)", "station holders"))
    blocks = []
    for match in TIME_RE.finditer(header):
        if match.group(0) not in blocks:
            blocks.append(match.group(0))
    blocks.sort(key=lambda block: tuple(int(part) for part in block.split(":")))
    if len(set(people)) != len(people) or len(set(stations)) != len(stations):
        raise HeaderError("duplicate names or stations in header")
    if set(people) & set(stations):
        raise HeaderError("a name is also a station")
    if len(blocks) != len(people):
        raise HeaderError("number of blocks differs from number of people")
    if len(holders) != len(stations) or len(set(holders)) != len(holders) or not set(holders) <= set(people):
        raise HeaderError("station holders do not match the stations and people")
    for field, value in (("n_staff", len(people)), ("n_stations", len(stations))):
        if isinstance(raw.get(field), int) and raw[field] != value:
            raise HeaderError(f"{field} does not match the header")
    lines = [line.strip() for line in body.split("\n") if line.strip()]
    return {"id": item_id, "people": people, "blocks": blocks, "stations": stations,
            "holders": holders, "lines": lines}


# Prompt construction

CORE_PROMPT = """You translate lines of shift notes into formal constraints.

The input lists the people, the time blocks in order, the stations and the station holders. These rules always apply and must not be output: every person works exactly one block, every block has exactly one person, and every station has exactly one holder chosen from the station holders.

Output format: exactly one output line per input line, in the same order, and nothing else:
L<number>: <relation>
L<number>: <relation> ; <relation>
L<number>: none
L<number>: unclear
Do not explain, quote the input, add notes or reason aloud.

Relations. X, Y, A and B are a person or a station. To mean whoever holds a station, write the station name itself, for example before(packing, Tomas). Never write ?, placeholders or invented names. P is a person, S is a station, T is a block time.
at(X, T) - X works block T.
not_at(X, T) - X does not work block T.
station(P, S) - P holds station S.
not_station(P, S) - P does not hold station S.
before(X, Y) - X works an earlier block than Y, with any gap.
later(X, Y) - X works a later block than Y, with any gap.
next(X, Y) - Y works the block immediately after X's block, so X is the one who works first.
between(X, A, B) - X, the first argument, works a block somewhere between the blocks of A and B, in either order, not necessarily next to either.
unclear - the line states a fact about the current rota that no relation above can express.
No other relation names exist.

Rules:
- Translate only what the line itself states. Never add facts from other lines, from the input lists or from guesses.
- Copy names, stations and times exactly as they appear in the input lists.
- Hedges such as "I think" or "as far as I know", and lead-ins such as "To repeat:", do not weaken a statement.
- For before and later, use the word that matches the line: if the line says someone is earlier, first or before, use before; if it says someone is later or after, use later. Keep the names in the order the line mentions them.
- Use next only when the line says the two blocks are consecutive, with nothing in between. In next, the person who works first is always the first argument.
- A line that puts X between A and B in either order, or after one of them and before the other, is exactly one relation: between(X, A, B). Being between two people never means being next to them.
- Relations joined by " ; " must all be true at the same time. Never write both orders of the same pair. Use " ; " only when a line states several separate facts.
- A line that says nothing about which block someone works or which station someone holds is none."""

GUIDANCE_PROMPT = """Only statements about the current rota count. These are none, even when they mention a person, a block or a station:
- wishes, requests or applications, whether granted or refused;
- earlier or previous arrangements;
- hypotheticals and what would or could have happened;
- questions, doubts or discussions that were never settled."""

EXAMPLE_SETTING = ("Example setting. People: Alice, Bob, Carla, Dev, Erin. Blocks in time order: 08:00, 10:00, "
                   "12:00, 14:00, 16:00. Stations: loading, sorting, testing. Station holders: Alice, Bob, Carla. "
                   "Each example line is independent of the others.")

CORE_EXAMPLES = [
    ("Carla is rostered for 12:00.", "at(Carla, 12:00)"),
    ("Dev can't do the 08:00 slot.", "not_at(Dev, 08:00)"),
    ("Sorting falls to Bob.", "station(Bob, sorting)"),
    ("Alice won't be on loading.", "not_station(Alice, loading)"),
    ("Erin starts some time after Bob finishes.", "later(Erin, Bob)"),
    ("Whoever runs testing clocks in ahead of Dev.", "before(testing, Dev)"),
    ("Dev turns up later than the loading holder.", "later(Dev, loading)"),
    ("The sorting desk is staffed before Erin arrives.", "before(sorting, Erin)"),
    ("Alice comes on the moment Carla's slot ends.", "next(Carla, Alice)"),
    ("Bob works, then Carla takes the very next slot.", "next(Bob, Carla)"),
    ("Erin takes over from Dev with no gap.", "next(Dev, Erin)"),
    ("Bob's slot lands somewhere between Erin's and Alice's.", "between(Bob, Erin, Alice)"),
    ("Whichever of Alice and Dev starts first, Erin comes after that one and before the other.",
     "between(Erin, Alice, Dev)"),
    ("To repeat: I think 16:00 belongs to Dev.", "at(Dev, 16:00)"),
    ("Carla runs testing and works 10:00.", "station(Carla, testing) ; at(Carla, 10:00)"),
    ("Dev and Alice share a locker.", "none"),
]

GUIDANCE_EXAMPLES = [
    ("Carla hoped for 08:00, but it went to someone else.", "none"),
    ("Two weeks ago Bob was the one on loading.", "none"),
    ("If Erin had worked 14:00, lunch cover would have been easier.", "none"),
    ("Whether Alice works 10:00 was never settled.", "none"),
]


def system_prompt(guidance=True):
    examples = CORE_EXAMPLES + (GUIDANCE_EXAMPLES if guidance else [])
    inputs = "\n".join(f"L{i}: {line}" for i, (line, _) in enumerate(examples, 1))
    outputs = "\n".join(f"L{i}: {result}" for i, (_, result) in enumerate(examples, 1))
    sections = [CORE_PROMPT]
    if guidance:
        sections.append(GUIDANCE_PROMPT)
    sections.append(f"{EXAMPLE_SETTING}\n\nExample input lines:\n{inputs}\n\nExample output:\n{outputs}")
    return "\n\n".join(sections)


def prompt_hash(guidance=True):
    return hashlib.sha256(system_prompt(guidance).encode("utf-8")).hexdigest()[:12]


def build_messages(item, guidance=True):
    count = len(item["lines"])
    listing = [
        f"People: {', '.join(item['people'])}",
        f"Blocks in time order: {', '.join(item['blocks'])}",
        f"Stations: {', '.join(item['stations'])}",
        f"Station holders: {', '.join(item['holders'])} (everyone else holds no station)",
        "",
        "Lines:",
    ]
    listing += [f"L{i}: {line}" for i, line in enumerate(item["lines"], 1)]
    listing += ["", f"There are {count} lines. Write exactly {count} output lines, L1 to L{count}, and nothing else."]
    return [{"role": "system", "content": system_prompt(guidance)},
            {"role": "user", "content": "\n".join(listing)}]


# Model-output parsing and validation

LINE_RE = re.compile(r"^[\s*>`#\-]*L\s*(\d+)[*`\s]*[:.)\]\-=]\s*(.*)$", re.IGNORECASE)
RELATION_RE = re.compile(r"^([A-Za-z_]+)\s*\((.*)\)$")
SIGNATURES = {
    "at": ("ref", "time"),
    "not_at": ("ref", "time"),
    "station": ("person", "station"),
    "not_station": ("person", "station"),
    "before": ("ref", "ref"),
    "later": ("ref", "ref"),
    "next": ("ref", "ref"),
    "between": ("ref", "ref", "ref"),
}
COMMENT_MARKERS = ("#", "//", " (", " — ", " -- ", " or ")


def resolve_argument(argument, role, item, normalise):
    value = argument.strip().strip("'\"`").strip()
    candidates = {"person": item["people"], "station": item["stations"], "time": item["blocks"],
                  "ref": item["people"] + item["stations"]}[role]
    if value in candidates:
        return value
    if not normalise:
        return None
    if role == "time":
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        if not match:
            return None
        wanted = (int(match.group(1)), int(match.group(2)))
        hits = [block for block in candidates if tuple(int(part) for part in block.split(":")) == wanted]
    else:
        hits = [candidate for candidate in candidates if candidate.casefold() == value.casefold()]
        if not hits:
            words = {word.casefold() for word in re.findall(r"[^\W\d_]+", value)}
            hits = [candidate for candidate in candidates if candidate.casefold() in words]
    return hits[0] if len(hits) == 1 else None


def parse_relation(text, item, normalise):
    match = RELATION_RE.match(text)
    if not match:
        return None
    name = match.group(1).lower()
    arguments = match.group(2).split(",")
    if (normalise and name in ("at", "not_at") and len(arguments) == 2
            and resolve_argument(arguments[1], "time", item, True) is None
            and resolve_argument(arguments[1], "station", item, True) is not None):
        name = "station" if name == "at" else "not_station"
    signature = SIGNATURES.get(name)
    if signature is None or len(arguments) != len(signature):
        return None
    resolved = [resolve_argument(arg, role, item, normalise) for arg, role in zip(arguments, signature)]
    if any(value is None for value in resolved):
        return None
    refs = [value for value, role in zip(resolved, signature) if role == "ref"]
    if len(set(refs)) != len(refs):
        return None
    if name == "later":
        return ("before", resolved[1], resolved[0])
    return (name, *resolved)


def clean_content(content):
    for arrow in ("→", "=>", "->"):
        if arrow in content:
            content = content.rsplit(arrow, 1)[1]
    content = re.sub(r"(\w)\s+\(", r"\1(", content)
    for marker in COMMENT_MARKERS:
        content = content.split(marker, 1)[0]
    return content


def parse_line_content(content, item, normalise):
    if normalise:
        content = clean_content(content)
    parts = [part.strip().strip("`*").rstrip(".").strip() for part in content.split(";")]
    parts = [part for part in parts if part]
    if len(parts) == 1 and re.match(r"none\b", parts[0], re.IGNORECASE):
        return {"status": "none", "relations": []}
    if len(parts) == 1 and re.match(r"unclear\b", parts[0], re.IGNORECASE):
        return {"status": "unclear", "relations": []}
    relations = [parse_relation(part, item, normalise) for part in parts]
    if not relations or any(relation is None for relation in relations):
        return {"status": "invalid", "relations": []}
    return {"status": "ok", "relations": relations}


def parse_output(text, item, normalise=True):
    count = len(item["lines"])
    parsed = [{"status": "missing", "relations": []} for _ in range(count)]
    for raw in (text or "").splitlines():
        match = LINE_RE.match(raw)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if not 0 <= index < count or parsed[index]["status"] not in ("missing", "invalid"):
            continue
        result = parse_line_content(match.group(2), item, normalise)
        if parsed[index]["status"] == "missing" or result["status"] != "invalid":
            parsed[index] = result
    return parsed


# Constraint evaluation and exhaustive enumeration

def build_space(item):
    space = []
    for order in itertools.permutations(range(len(item["people"]))):
        block_of = dict(zip(item["people"], order))
        for chosen in itertools.permutations(item["holders"]):
            space.append((block_of, dict(zip(item["stations"], chosen))))
    return space


def prepare(item):
    space = build_space(item)
    if not space:
        raise HeaderError("header rules admit no assignment")
    return {"item": item, "space": space, "full": (1 << len(space)) - 1,
            "block_index": {block: i for i, block in enumerate(item["blocks"])}, "cache": {}}


def position(assignment, ref):
    block_of, holder_of = assignment
    return block_of[holder_of.get(ref, ref)]


def relation_holds(relation, assignment, block_index):
    name, args = relation[0], relation[1:]
    holder_of = assignment[1]
    if name == "at":
        return position(assignment, args[0]) == block_index[args[1]]
    if name == "not_at":
        return position(assignment, args[0]) != block_index[args[1]]
    if name == "station":
        return holder_of[args[1]] == args[0]
    if name == "not_station":
        return holder_of[args[1]] != args[0]
    if name == "before":
        return position(assignment, args[0]) < position(assignment, args[1])
    if name == "next":
        return position(assignment, args[1]) == position(assignment, args[0]) + 1
    if name == "between":
        x, a, b = (position(assignment, ref) for ref in args)
        return min(a, b) < x < max(a, b)
    raise ValueError(f"unknown relation {name}")


def relation_mask(ctx, relation):
    cache = ctx["cache"]
    if relation not in cache:
        mask = 0
        for i, assignment in enumerate(ctx["space"]):
            if relation_holds(relation, assignment, ctx["block_index"]):
                mask |= 1 << i
        cache[relation] = mask
    return cache[relation]


def analyse_sample(ctx, text, normalise=True):
    parsed = parse_output(text, ctx["item"], normalise)
    statuses, masks = [], []
    for line in parsed:
        status, mask = line["status"], None
        if status == "ok":
            mask = ctx["full"]
            for relation in line["relations"]:
                mask &= relation_mask(ctx, relation)
            if mask == 0:
                status, mask = "contradictory", None
        elif status == "none":
            mask = ctx["full"]
        statuses.append(status)
        masks.append(mask)
    return {"statuses": statuses, "relations": [line["relations"] for line in parsed],
            "masks": masks, "usable": any(mask is not None for mask in masks)}


def and_all(masks, indices, full):
    result = full
    for i in indices:
        result &= masks[i]
    return result


def solution_count(ctx, masks):
    return and_all(masks, range(len(masks)), ctx["full"]).bit_count()


# Minimal conflict detection

def is_minimal_conflict(masks, core, full):
    if and_all(masks, core, full):
        return False
    return all(and_all(masks, [k for k in core if k != j], full) for j in core)


def find_core(masks, full, support=None, rule="smallest"):
    active = [j for j, mask in enumerate(masks) if mask != full]
    if any(masks[j] == 0 for j in active):
        raise RuntimeError("a single line is unsatisfiable on its own and should have been rejected")
    if and_all(masks, active, full):
        raise RuntimeError("line constraints are satisfiable, so there is no conflict")
    if rule == "all":
        return active
    if rule == "deletion":
        core = list(active)
        for j in active:
            trial = [k for k in core if k != j]
            if not and_all(masks, trial, full):
                core = trial
    elif rule == "smallest":
        weight = support or [1] * len(masks)
        core = None
        for size in range(1, len(active) + 1):
            found = [combo for combo in itertools.combinations(active, size) if not and_all(masks, combo, full)]
            if found:
                core = list(min(found, key=lambda combo: (-min(weight[j] for j in combo), combo)))
                break
    else:
        raise ValueError(f"unknown core rule {rule}")
    if not is_minimal_conflict(masks, core, full):
        raise RuntimeError("conflict set failed the minimality check")
    return core


# Answer formatting

def format_assignment(ctx, assignment):
    block_of, holder_of = assignment
    station_of = {person: station for station, person in holder_of.items()}
    result = {}
    for person in ctx["item"]["people"]:
        entry = {"block": ctx["item"]["blocks"][block_of[person]]}
        if person in station_of:
            entry["station"] = station_of[person]
        result[person] = entry
    return result


def decide(ctx, masks, support=None, core_rule="smallest"):
    full = ctx["full"]
    combined = and_all(masks, range(len(masks)), full)
    if combined:
        people = ctx["item"]["people"]
        found = [format_assignment(ctx, ctx["space"][i]) for i in range(len(ctx["space"])) if combined >> i & 1]
        if len(found) == 1:
            return {"case": "unique", "assignment": found[0]}
        found.sort(key=lambda a: [(a[p]["block"], a[p].get("station", "")) for p in people])
        return {"case": "ambiguous", "assignments": found}
    core = find_core(masks, full, support, core_rule)
    return {"case": "inconsistent", "conflicts": [ctx["item"]["lines"][j] for j in core]}


def describe(answer):
    if answer["case"] == "unique":
        return {"declared": "unique", "solutions": 1, "core_size": None}
    if answer["case"] == "ambiguous":
        return {"declared": "ambiguous", "solutions": len(answer["assignments"]), "core_size": None}
    return {"declared": "inconsistent", "solutions": 0, "core_size": len(answer["conflicts"])}


def canonical(answer):
    return json.dumps(answer, sort_keys=True, ensure_ascii=False)


# Aggregation policies

def line_vote(analyses, ctx):
    masks, support = [], []
    for j in range(len(ctx["item"]["lines"])):
        votes = [a["masks"][j] for a in analyses if a["masks"][j] is not None]
        if not votes:
            masks.append(ctx["full"])
            support.append(0)
            continue
        counts = Counter(votes)
        best = max(counts.values())
        masks.append(next(vote for vote in votes if counts[vote] == best))
        support.append(best)
    return masks, support


def sample_answer(ctx, analysis, core_rule="smallest"):
    masks, support = line_vote([analysis], ctx)
    return decide(ctx, masks, support, core_rule)


def majority(answers):
    keys = [canonical(answer) for answer in answers]
    counts = Counter(keys)
    best = max(counts.values())
    return answers[next(i for i, key in enumerate(keys) if counts[key] == best)]


def run_policy(policy, cap, draw, ctx, normalise=True, core_rule="smallest"):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    analyses, failed, agreed = [], 0, None

    def take():
        nonlocal failed
        text = draw(len(analyses) + failed)
        if text is None:
            failed += 1
            return None
        analysis = analyse_sample(ctx, text, normalise)
        analyses.append(analysis)
        return analysis

    limit = 1 if policy == "single" else cap
    if policy in ("single", "fixed_vote", "answer_vote"):
        for _ in range(limit):
            take()
    elif policy == "agreement":
        seen = Counter()
        for _ in range(limit):
            analysis = take()
            if analysis and analysis["usable"]:
                answer = sample_answer(ctx, analysis, core_rule)
                seen[canonical(answer)] += 1
                if seen[canonical(answer)] >= 2:
                    agreed = answer
                    break
    else:
        for _ in range(limit):
            take()
            usable = [a for a in analyses if a["usable"]]
            if usable:
                masks, support = line_vote(usable, ctx)
                if all(s > 0 for s in support) and solution_count(ctx, masks) <= MAX_PLAUSIBLE_SOLUTIONS:
                    break

    result = {"calls": len(analyses) + failed, "failed_calls": failed,
              "line_status_counts": dict(Counter(s for a in analyses for s in a["statuses"])),
              "line_statuses": [a["statuses"] for a in analyses]}
    usable = [a for a in analyses if a["usable"]]
    if not analyses:
        return {**result, "outcome": "failed", "category": "api_error", "answer": None}
    if not usable:
        return {**result, "outcome": "failed", "category": "no_usable_extraction", "answer": None}
    masks, support = line_vote(usable, ctx)
    if agreed is not None:
        answer = agreed
    elif policy == "answer_vote":
        answer = majority([sample_answer(ctx, a, core_rule) for a in usable])
    else:
        answer = decide(ctx, masks, support, core_rule)
    unusable = [j + 1 for j, s in enumerate(support) if s == 0]
    degraded = failed > 0 or bool(unusable)
    return {**result, **describe(answer), "unusable_lines": unusable, "outcome": "answered",
            "category": "degraded" if degraded else "ok", "answer": answer}


# Budget enforcement and API calls

class CallBudget:
    def __init__(self, cap):
        self.cap = cap
        self.used = 0

    def take(self):
        if self.used >= self.cap:
            raise BudgetExceeded(f"call {self.used + 1} would exceed the cap of {self.cap}")
        self.used += 1


def request_completion(client, item_id, messages, budget):
    budget.take()
    started = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            extra_headers={"X-Item-Id": item_id},
            extra_body={"reasoning": {"enabled": False}},
        )
        choice = response.choices[0]
        return {"text": choice.message.content or "", "error": None,
                "finish_reason": getattr(choice, "finish_reason", None),
                "model": getattr(response, "model", None), "latency_s": round(time.monotonic() - started, 2)}
    except Exception as exc:
        return {"text": None, "error": type(exc).__name__, "finish_reason": None, "model": None,
                "latency_s": round(time.monotonic() - started, 2)}


def process_item(client, raw, budget_name, policy):
    cap = BUDGETS[budget_name]
    record = {"id": raw["id"], "policy": policy, "calls": 0, "call_errors": [], "raw_outputs": []}
    try:
        ctx = prepare(parse_item(raw))
    except HeaderError as exc:
        record.update(outcome="failed", category="header_error", detail=str(exc), answer=None)
        return record
    messages = build_messages(ctx["item"])
    budget = CallBudget(cap)

    def draw(index):
        reply = request_completion(client, raw["id"], messages, budget)
        record["raw_outputs"].append({"sample": index, **reply})
        if reply["error"]:
            record["call_errors"].append(reply["error"])
        return reply["text"]

    try:
        record.update(run_policy(policy, cap, draw, ctx))
    except BudgetExceeded:
        raise
    except Exception as exc:
        record.update(outcome="failed", category="internal_error", detail=f"{type(exc).__name__}: {exc}", answer=None)
    record["calls"] = budget.used
    if not 1 <= budget.used <= cap or (budget_name == "1x" and budget.used != 1):
        raise BudgetExceeded(f"{raw['id']} used {budget.used} calls under budget {budget_name}")
    return record


# CLI orchestration

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="run", description="Answer shift-note items with Granite extraction and exhaustive solving.")
    parser.add_argument("items", help="path to items.json")
    parser.add_argument("--budget", required=True, choices=list(BUDGETS), help="model calls allowed per item")
    parser.add_argument("--out", required=True, help="path for the answers JSON")
    return parser.parse_args(argv)


def load_items(path):
    try:
        with open(path, encoding="utf-8") as handle:
            items = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read items file {path}: {exc}") from None
    if not isinstance(items, list) or not all(
            isinstance(i, dict) and isinstance(i.get("id"), str) and isinstance(i.get("text"), str) for i in items):
        raise ConfigError("items file must be a list of objects with string 'id' and 'text'")
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        raise ConfigError("items file contains duplicate ids")
    return items


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    args = parse_args(argv)
    try:
        items = load_items(args.items)
        client = make_client()
        workers = worker_count()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    policy = SHIPPED_POLICY[args.budget]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda raw: process_item(client, raw, args.budget, policy), items))
    answers = {r["id"]: r["answer"] for r in results if r["outcome"] == "answered"}
    failed = [r["id"] for r in results if r["outcome"] == "failed"]
    degraded = sum(1 for r in results if r.get("category") == "degraded")
    calls = sum(r["calls"] for r in results)
    write_json_atomic(args.out, answers)
    status_path = Path(str(args.out) + ".status.json")
    write_json_atomic(status_path, {
        "budget": args.budget, "policy": policy, "model": MODEL_ID, "prompt_hash": prompt_hash(),
        "items": len(items), "answered": len(answers), "degraded": degraded, "failed": len(failed),
        "calls": calls, "results": [{k: v for k, v in r.items() if k != "answer"} for r in results],
    })
    print(f"{len(answers)}/{len(items)} items answered ({degraded} degraded), {len(failed)} failed, "
          f"{calls} model calls. Status: {status_path}", file=sys.stderr)
    if failed:
        print(f"INCOMPLETE: {len(failed)} of {len(items)} items not answered: {', '.join(failed)}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())