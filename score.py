#!/usr/bin/env python3
"""
Scorer.

The headline metric is `macro_exact_match`: the mean of the three per-case
rates, one for unique, one for ambiguous, one for inconsistent.  Micro
`exact_match` -- the plain fraction of items fully resolved -- is reported
beside it as a diagnostic and is NOT the ranking number.

Why: macro caps a single-kind strategy at a third whatever the mix happens to
be.  A solver that answers every unique item and declares everything unique
takes the whole unique share on the micro figure while having no representation
of ambiguity or contradiction at all -- 80.0% on an 80/10/10 set.  This set is
20/20/20, so the two numbers coincide here and that strategy scores 33% on
both; macro is the ranking number because the coincidence is a property of this
mix and not of the metric.

cell_accuracy is reported for diagnosis only and is NOT a ranking metric --
partial credit rewards guessing under a permutation constraint.

An item is exact_match only if your submission gets its case right AND gives the
right content for that case:

  unique        the single consistent assignment is correct
  ambiguous     EVERY consistent assignment is given and no extra ones. There
                are between 2 and 4 of them and you are not told how many;
                order does not matter, cardinality does.
  inconsistent  the item is declared inconsistent AND a conflicting set is
                cited: the cited statements cannot all be true at once, and
                dropping any one of them leaves a set that can. That is the
                definition of a minimal conflicting set, and it is the whole
                rule -- there is no separate list of "the" right answer to
                match. An item usually has more than one such set and every one
                of them is correct. Citing extra statements is wrong (drop one
                and it is still contradictory), and so is citing too few (they
                no longer contradict). Declaring the case and citing nothing is
                not a correct answer to an inconsistent item, any more than
                declaring "unique" and giving no assignment is a correct answer
                to a unique one.

Also reported, and read alongside exact_match rather than folded into it.
Neither of the two partial-credit figures is ever ranked on; they say where a
submission is on the cliff, which the headline deliberately does not:

  ambiguous_partial          on ambiguous items: you declared ambiguous and
                             every assignment you gave is a real one, but you
                             may not have given them all. Exact answers count
                             in it too, so it is an upper bound on the
                             ambiguous rate rather than a disjoint bucket.
  inconsistent_label_only    on inconsistent items: you called the case right
                             and the cited set is not a minimal conflicting
                             one -- too many statements, too few, or not
                             statements from the notes.


  confusion             keyed by the TRUE case, each holding a count of what
                        was declared for it: 3x3 plus an "unparsed" column. The errors have direction and a scalar hides
                        it: over-declaring ambiguous is a system failing
                        honestly, and only ever declaring unique means there is
                        no formal representation underneath at all.
  conflict_containment  null on your copy of the key, which does not record
                        which statement was added to break each item. It is
                        computed when we grade, and it asks whether your cited
                        set CONTAINS that statement.
                        Containment, not equality: a real minimal unsat core
                        legitimately also cites the statements it contradicts.
                        The denominator is every inconsistent item in the set,
                        so silence scores the same as a wrong citation, and
                        conflict_coverage says how many you attempted.
                        Saturable by citing every line, so it is reported with
                        your mean cited-set size and mean cited length against
                        the true core's, and it only ranks if neither is more
                        than twice the core's AND you attempted every
                        inconsistent item -- citing well on 6 of 20 and staying
                        silent on the rest is not a better score than citing
                        well on 20 of 20.

Submission forms, keyed by item id (see example_answers.json):

  {"case": "unique",       "assignment":  {"<name>": {"block": "..", "station": ".."}, ...}}
  {"case": "ambiguous",    "assignments": [ {...}, {...} ]}
  {"case": "inconsistent", "conflicts":   ["<statement>", ...]}

A bare assignment object with no "case" field is read as a unique claim.

Where an item puts only some of the staff on a station, give "station" only for
those people. A station for someone the notes put on no station is ignored, not
penalised; a missing person always is.

conflict_containment is the diagnostic half of the inconsistent branch: it
asks only whether the injected statement is in there, with no bound on what
else is, so it is read beside the cited-set sizes and never on its own.

usage: python3 score.py visible_key.json <your answers>.json [items.json]

items.json is optional and is found automatically if it sits next to the key.
It is used for one thing: a cited statement has to actually occur in the item
it is cited from.
"""
import json, os, sys

CASES = ("unique", "ambiguous", "inconsistent")


def norm(a):
    """Canonical comparable form of one assignment."""
    if not isinstance(a, dict):
        return None
    out = {}
    for person, fields in a.items():
        if not isinstance(fields, dict):
            return None
        out[str(person)] = tuple(sorted((str(k), str(v)) for k, v in fields.items()))
    return out


def match(truth, guess):
    """Does a submitted assignment match the key's?

    Only the fields the key names are compared. Where an item puts just some of
    the staff on a station, the others have no station to get right, so an
    unconstrained station is neither required nor penalised if volunteered.
    Everyone on the rota must still be present.
    """
    if not isinstance(guess, dict) or set(guess) != set(truth):
        return False
    for person, fields in truth.items():
        g = dict(guess[person])
        if any(g.get(f) != v for f, v in fields):
            return False
    return True


def read_sub(got):
    """(case, payload) from a submission entry, or (None, None) if unparsable."""
    if not isinstance(got, dict):
        return None, None
    case = got.get("case")
    if case is None:
        n = norm(got)                       # legacy bare form == unique claim
        return ("unique", n) if n else (None, None)
    if case not in CASES:
        return None, None
    # "assignment" and a one-element "assignments" say the same thing, and
    # "conflicts" and "conflict_lines" do too. Accepting both shapes costs
    # nothing and means answers.json -- which is the key -- also reads as a
    # correct submission, so you can check your harness end to end against it.
    if case == "unique":
        a = got.get("assignment")
        if a is None:
            lst = got.get("assignments")
            a = lst[0] if isinstance(lst, list) and len(lst) == 1 else None
        return case, norm(a)
    if case == "ambiguous":
        lst = got.get("assignments")
        if lst is None and got.get("assignment") is not None:
            lst = [got["assignment"]]
        if not isinstance(lst, list):
            return case, None
        normed = [norm(x) for x in lst]
        return case, (normed if all(n is not None for n in normed) else None)
    cs = got.get("conflicts")
    if cs is None:
        cs = got.get("conflict_lines")
    if cs is None:
        # the key's own shape, so `score.py visible_key.json visible_key.json`
        # scores 1.0 and you can check your harness end to end against it. Any
        # one of the recorded sets is a correct answer; the first will do.
        sets = got.get("conflict_line_sets")
        cs = sets[0] if isinstance(sets, list) and sets else []
    return case, ([str(x) for x in cs] if isinstance(cs, list) else [])


# Hedge openings. A citation that drops the hedge is still a citation of the
# statement, so both sides are normalised past them before anything is compared.
HEDGE_OPENERS = (
    "i'm fairly sure ", "from what i recall, ", "as far as i know, ",
    "i believe ", "going off the roster, ", "if memory serves, ",
    "my recollection is that ", "unless i have this wrong, ",
    "speaking from memory, ", "i'm reasonably confident ",
)


def squash(s):
    """Whitespace-collapsed, case-folded form."""
    return " ".join(str(s).split()).casefold()


def squash_stmt(s):
    """The same, with a restatement prefix and a hedge opening taken off.

    Applied to citations and to the statement they are compared against, never
    to the whole item text -- the item's header has a colon in it.
    """
    s = squash(s)
    head, sep, tail = s.partition(": ")
    if sep and tail and len(head) < 60:
        s = tail
    for h in HEDGE_OPENERS:
        if s.startswith(h):
            return s[len(h):]
    return s


# A citation may be longer than the statement -- quoting a little surrounding
# context is normal -- but a citation three times its length is not a citation
# of it.
MAX_ENVELOPE = 3.0
# A floor on the citation's length as a fraction of the statement's. It is
# implied by the containment direction below -- `want in got` already forces
# len(got) >= len(want) -- and is kept only so that a short citation fails at
# the length check rather than three lines further down. It is NOT a licence to
# quote 60% of a statement; the brief no longer says it is.
MIN_FRAGMENT = 0.6


def cites(got, want, text=None):
    """Does this one citation pick out this one statement?

    Three conditions, all required:
      - the citation contains the statement (after hedge and case normalising);
      - it is not more than MAX_ENVELOPE times its length;
      - it is at least MIN_FRAGMENT of its length, and it actually occurs in
        the item it is cited from, when the item text is available.
    """
    if not got or not want:
        return False
    # Note this bound is implied by `want in got` below, which forces
    # len(got) >= len(want): the one-character citation is stopped by the
    # containment direction, not by this line. Kept because MIN_FRAGMENT is the
    # live floor on the legacy path in core_ok, and because a citation shorter
    # than the statement should fail here rather than three lines further down.
    if len(got) < MIN_FRAGMENT * len(want):
        return False
    if text is not None and got not in text:
        return False
    return want in got and len(got) <= MAX_ENVELOPE * len(want)


def smallest_core(k):
    """The shortest minimal conflicting set the key records for this item.

    Used only for the size diagnostics -- a cited set is scored against every
    recorded set, not this one -- and the shortest is the right reference
    because it is the one a submission is compared against when asking whether
    its citations are systematically bloated.
    """
    cores = k.get("conflict_line_sets") or []
    return min(cores, key=len) if cores else []


def contains_injected(cited, injected, text=None):
    """Does the cited set include the injected statement?"""
    want = squash_stmt(injected)
    return any(cites(squash_stmt(c), want, text) for c in cited)


def matches_core(cited, core, text=None, body=None):
    """Is this cited set exactly this core, one citation per statement?

    Element-wise under the same hedge/case normalisation the rest of the file
    uses, so quoting a line without its hedge still counts, and no citation may
    stand in for two statements.
    """
    if len(cited) != len(core):
        return False
    cs = [squash_stmt(c) for c in cited]
    # A citation may quote AT MOST ONE statement of the notes. The envelope
    # bound alone did not enforce that: it is checked against the one statement
    # a citation is matched to, so a two-line span that also swallowed a
    # padding or filler line stayed inside 3x and scored as a citation of the
    # first. Every cited set could then be padded with inert lines and still
    # read as minimal -- 18 of 20 inconsistent items in the shipped set were
    # answerable that way, which is the opposite of the rule the brief states.
    #
    # Counted over DISTINCT normalised lines: a restatement normalises to the
    # same string as the line it restates, so counting occurrences would reject
    # an honest citation of either one.
    if body:
        for c in cs:
            if len({l for l in body if l and l in c}) > 1:
                return False
    # Two holes, both measured. Nothing required the citations to be DISTINCT,
    # and MAX_ENVELOPE is checked against the one statement a citation is
    # matched to -- so one contiguous span swallowing two core statements,
    # submitted twice, read as exact set equality with a two-statement core.
    # On the balanced set 243 of 399 inconsistent items could be answered that
    # way, several of the spans including padding lines that assert nothing.
    if len(set(cs)) != len(cs):
        return False
    ws = [squash_stmt(w) for w in core]
    for c in cs:
        if sum(1 for w in ws if w in c) > 1:
            return False
    left = list(core)
    for c in cs:
        hit = next((w for w in left if cites(c, squash_stmt(w), text)), None)
        if hit is None:
            return False
        left.remove(hit)
    return not left


def core_ok(cited, cores, text=None, body=None):
    """Is the cited set a minimal conflicting set of this item's statements?

    The rule the brief states is a property of the cited set, not a lookup:
    the statements cannot all hold at once, and dropping any one of them leaves
    a set that can.  Every set with that property is a correct answer, and an
    item routinely has several.

    It is checked against `conflict_line_sets`, which records every one of them.
    That is not a weaker check -- it is the same check.  Both queries the rule
    makes are decided entirely by which sets have the property: a cited set is
    contradictory exactly when it contains one of them, and it is minimal
    exactly when it IS one of them.  So enumerating them and evaluating the rule
    are the same computation, and enumerating is the one that does not require
    shipping a parser or a per-line constraint map with the key.

    The inconsistent branch used to be scored on the label alone -- `case ==
    "inconsistent"` was the whole test -- while unique needed the right
    assignment and ambiguous the complete set.  Declaring every item
    inconsistent therefore took the entire stratum, a third of the macro
    headline, for nothing.
    """
    if not cores or not cited:
        return False
    return any(matches_core(cited, c, text, body) for c in cores)


def score(key, sub, items=None):
    # A submission that is not an id->answer mapping scores zero rather than
    # crashing the grading run: a list, a string or null is a wrong answer for
    # every item, not a reason to lose the whole batch.
    if not isinstance(sub, dict):
        sub = {}
    # id -> normalised item text, so a citation can be checked against the notes
    # it claims to quote. Optional: without it the length and envelope bounds
    # still apply, only the "did you make this up" check is skipped.
    texts, bodies = {}, {}
    if items:
        for it in items:
            if isinstance(it, dict) and "id" in it and "text" in it:
                texts[it["id"]] = squash(it["text"])
                # every body line, normalised the way a citation is, so a
                # citation can be checked for quoting more than one of them
                bodies[it["id"]] = [squash_stmt(l) for l
                                    in it["text"].split("\n")[1:] if l.strip()]
    per_case = {c: {"n": 0, "correct": 0} for c in CASES}
    confusion = {t: {d: 0 for d in CASES + ("unparsed",)} for t in CASES}
    cells_right = cells_total = missing = exact = 0
    cited = attempted = inj_known = 0
    amb_partial = inc_label_only = 0
    cited_sizes, core_sizes, cited_chars, core_chars = [], [], [], []
    per_item = {}

    for iid, k in key.items():
        case = k["case"]
        per_case[case]["n"] += 1
        got_case, payload = read_sub(sub.get(iid))
        confusion[case][got_case or "unparsed"] += 1
        if got_case is None:
            missing += 1
            if case == "unique":
                cells_total += sum(len(v) for v in k["assignments"][0].values())
            if case == "inconsistent":
                # counted, and counted as a miss. The denominator is every
                # inconsistent item in the set, not every one you attempted:
                # citing correctly on 2 of 6 and staying silent on the rest used
                # to read 1.00, indistinguishable from full coverage.
                core = smallest_core(k)
                core_sizes.append(len(core))
                core_chars.append(sum(len(squash(c)) for c in core))
                cited_sizes.append(0)
                cited_chars.append(0)
            per_item[iid] = 0.0
            continue

        ok = False
        if case == "unique":
            truth = norm(k["assignments"][0])
            ok = got_case == "unique" and match(truth, payload)
            # cell diagnostic only: score the best single assignment on offer,
            # whatever case was claimed, so a wrong case call does not hide
            # whether the reasoning underneath was any good.
            if got_case == "unique":
                guess = payload or {}
            elif got_case == "ambiguous" and payload:
                guess = payload[0]
            else:
                guess = {}
            r = t = 0
            for person, fields in truth.items():
                g = dict(guess.get(person, ()))
                for field, val in fields:
                    t += 1
                    r += 1 if g.get(field) == val else 0
            cells_right += r
            cells_total += t
            per_item[iid] = r / t if t else 0.0
        elif case == "ambiguous":
            truth = [norm(a) for a in k["assignments"]]
            ok = (got_case == "ambiguous" and payload is not None
                  and len(payload) == len(truth)
                  and all(any(match(t, p) for p in payload) for t in truth)
                  and all(any(match(t, p) for t in truth) for p in payload))
            # diagnostic, never ranked: everything you gave is a real solution,
            # whether or not you gave them all. Exact answers are in it too.
            if (got_case == "ambiguous" and payload
                    and all(any(match(t, p) for t in truth) for p in payload)):
                amb_partial += 1
            per_item[iid] = 1.0 if ok else 0.0
        else:
            cores = k.get("conflict_line_sets") or []
            core = smallest_core(k)
            core_sizes.append(len(core))
            core_chars.append(sum(len(squash(c)) for c in core))
            got = (payload if (got_case == "inconsistent"
                               and isinstance(payload, list)) else [])
            cited_sizes.append(len(got))
            cited_chars.append(sum(len(squash(c)) for c in got))
            # containment is only computable when the key names the statement
            # that was injected, and the shipped key deliberately does not. It
            # stays available for internal grading, where the private record is
            # merged in; on a candidate's own key it reads None throughout.
            if k.get("injected_line"):
                if contains_injected(got, k["injected_line"], texts.get(iid)):
                    cited += 1
                inj_known += 1
            if got:
                attempted += 1
            ok = got_case == "inconsistent" and core_ok(got, cores,
                                                        texts.get(iid),
                                                        bodies.get(iid))
            if got_case == "inconsistent" and not ok:
                inc_label_only += 1
            per_item[iid] = 1.0 if ok else 0.0

        if ok:
            exact += 1
            per_case[case]["correct"] += 1

    n = len(key)
    n_inc = per_case["inconsistent"]["n"]
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0.0
    cited_mean, core_mean = mean(cited_sizes), mean(core_sizes)
    cited_ch, core_ch = mean(cited_chars), mean(core_chars)
    # the headline: each case worth a third, whatever the mix. Cases with no
    # items in the set are dropped rather than scored 0, so a set without an
    # ambiguous stratum is a two-case macro rather than an automatic 0.67 cap.
    rates = [per_case[c]["correct"] / per_case[c]["n"]
             for c in CASES if per_case[c]["n"]]
    return {
        "n_items": n,
        "macro_exact_match": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "per_case_rate": {c: (round(per_case[c]["correct"] / per_case[c]["n"], 4)
                              if per_case[c]["n"] else None) for c in CASES},
        "exact_match": round(exact / n, 4) if n else 0.0,
        "per_case": per_case,
        "confusion": confusion,
        "cell_accuracy": round(cells_right / cells_total, 4) if cells_total else 0.0,
        # Reported, never ranked on. See the module docstring.
        "ambiguous_partial": (round(amb_partial / per_case["ambiguous"]["n"], 4)
                              if per_case["ambiguous"]["n"] else None),
        "inconsistent_label_only": (round(inc_label_only / n_inc, 4)
                                    if n_inc else None),
        "conflict_containment": (round(cited / n_inc, 4)
                                 if n_inc and inj_known == n_inc else None),
        # how many of them you cited anything at all for, so a high containment
        # on two items out of six cannot be read as a high containment
        "conflict_coverage": round(attempted / n_inc, 4) if n_inc else None,
        "cited_mean": cited_mean,
        "core_mean": core_mean,
        "cited_chars_mean": cited_ch,
        "core_chars_mean": core_ch,
        # citing every line in the item saturates containment, so the number
        # only ranks when the cited sets are not systematically bloated -- in
        # count OR in length, because one very long citation is the same trick
        # ...and only when every inconsistent item was actually attempted:
        # the same citations on 6 of 20 items used to rank while on 20 of 20
        # they did not, so answering fewer scored better.
        "containment_rankable": (cited_mean <= 2 * core_mean
                                 and cited_ch <= 2 * core_ch
                                 and n_inc > 0 and attempted == n_inc
                                 and inj_known == n_inc),
        "missing_or_unparsed": missing,
        "per_item": per_item,
    }


def find_items(keypath, given=None):
    """items.json for this key: the one you named, or the one beside the key."""
    for p in (given, os.path.join(os.path.dirname(os.path.abspath(keypath)),
                                  "items.json")):
        if p and os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return None
    return None


if __name__ == "__main__":
    key = json.load(open(sys.argv[1]))
    sub = json.load(open(sys.argv[2]))
    res = score(key, sub, find_items(sys.argv[1],
                                     sys.argv[3] if len(sys.argv) > 3 else None))
    res.pop("per_item")
    print(json.dumps(res, indent=2))
