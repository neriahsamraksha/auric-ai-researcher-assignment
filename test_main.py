import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import main

ROOT = Path(__file__).resolve().parent
ITEMS = {item["id"]: item for item in json.loads((ROOT / "items.json").read_text(encoding="utf-8"))}

SYNTHETIC = {
    "id": "TEST-1",
    "text": ("Shift notes, Test. 4 staff on the rota: Ann, Ben, Cat, Dan. Blocks run 07:00, 09:00, 11:00, 13:00, "
             "one person per block, and each person works exactly one block. There are 1 stations, one person "
             "on each: intake. The people on a station are Ann; the rest are on no station.\n\n"
             "First line.\nSecond line.\nThird line.\nFourth line."),
}

CYCLE = "L1: before(Ann, Ben)\nL2: before(Ben, Cat)\nL3: before(Cat, Ann)\nL4: at(Dan, 13:00)"


def context(raw):
    return main.prepare(main.parse_item(raw))


def answer_for(raw, text, core_rule="smallest"):
    ctx = context(raw)
    return main.run_policy("single", 1, lambda _index: text, ctx, core_rule=core_rule), ctx


def parsed_lines(text, item_id="B2-001", normalise=True):
    return main.parse_output(text, main.parse_item(ITEMS[item_id]), normalise)


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model=main.MODEL_ID)


class SolverTests(unittest.TestCase):
    def test_between_alone_is_satisfiable(self):
        ctx = context(ITEMS["B2-001"])
        self.assertEqual(main.relation_mask(ctx, ("between", "Rohan", "Ayesha", "Tomas")).bit_count(), 240)

    def test_single_between_line_is_not_inconsistent(self):
        result, _ = answer_for(ITEMS["B2-001"], "L14: between(Rohan, Ayesha, Tomas)")
        self.assertEqual(result["outcome"], "answered")
        self.assertNotEqual(result["answer"]["case"], "inconsistent")

    def test_single_next_line_is_not_inconsistent(self):
        result, _ = answer_for(ITEMS["B2-002"], "L2: next(Nadia, Priya)")
        self.assertEqual(result["outcome"], "answered")
        self.assertNotEqual(result["answer"]["case"], "inconsistent")

    def test_self_contradictory_line_is_rejected(self):
        result, _ = answer_for(ITEMS["B2-001"], "L1: none\nL14: before(Ayesha, Rohan) ; before(Rohan, Ayesha)")
        self.assertEqual(result["line_statuses"][0][13], "contradictory")
        self.assertNotEqual(result["answer"]["case"], "inconsistent")
        self.assertEqual(result["category"], "degraded")

    def test_impossible_set_is_inconsistent_and_minimal(self):
        for rule in ("smallest", "deletion"):
            result, ctx = answer_for(SYNTHETIC, CYCLE, rule)
            self.assertEqual(result["answer"]["case"], "inconsistent")
            self.assertEqual(result["answer"]["conflicts"], ["First line.", "Second line.", "Third line."])
            masks, _ = main.line_vote([main.analyse_sample(ctx, CYCLE)], ctx)
            self.assertTrue(main.is_minimal_conflict(masks, [0, 1, 2], ctx["full"]))

    def test_two_line_conflict_is_cited_exactly_without_padding(self):
        text = "L1: at(Ann, 07:00)\nL2: not_at(Ann, 07:00)\nL3: before(Ben, Cat)\nL4: at(Dan, 13:00)"
        for rule in ("smallest", "deletion"):
            result, ctx = answer_for(SYNTHETIC, text, rule)
            self.assertEqual(result["answer"]["conflicts"], ["First line.", "Second line."])
            masks, _ = main.line_vote([main.analyse_sample(ctx, text)], ctx)
            self.assertTrue(main.is_minimal_conflict(masks, [0, 1], ctx["full"]))

    def test_model_output_changes_answer(self):
        first, _ = answer_for(SYNTHETIC, "L1: before(Ann, Ben)\nL2: before(Ben, Cat)\nL3: before(Cat, Dan)\nL4: none")
        second, _ = answer_for(SYNTHETIC, "L1: before(Ben, Ann)\nL2: before(Ann, Cat)\nL3: before(Cat, Dan)\nL4: none")
        self.assertEqual(first["answer"]["case"], "unique")
        self.assertNotEqual(main.canonical(first["answer"]), main.canonical(second["answer"]))


class ParserTests(unittest.TestCase):
    def test_later_becomes_reversed_before(self):
        self.assertEqual(parsed_lines("L2: later(Tomas, packing)")[1]["relations"], [("before", "packing", "Tomas")])

    def test_commentary_is_removed(self):
        text = ("L2: before(packing, Tomas)  # because packing is earlier\n"
                "L3: \"Meera is not on packing.\" -> not_station(Meera, packing)\n"
                "L4: next(Daniel, Rohan) or maybe not\n"
                "L5: not_at(Rohan, 11:00) (since he lobbied)")
        parsed = parsed_lines(text)
        self.assertEqual([parsed[i]["status"] for i in (1, 2, 3, 4)], ["ok", "ok", "ok", "ok"])
        self.assertEqual(parsed[2]["relations"], [("not_station", "Meera", "packing")])

    def test_station_placeholder_resolves_only_when_unambiguous(self):
        parsed = parsed_lines("L2: before(whoever has packing, Tomas)\nL3: station(?, packing)")
        self.assertEqual(parsed[1]["relations"], [("before", "packing", "Tomas")])
        self.assertEqual(parsed[2]["status"], "invalid")

    def test_station_given_as_time_becomes_station(self):
        self.assertEqual(parsed_lines("L4: at(Tomas, intake)")[3]["relations"], [("station", "Tomas", "intake")])

    def test_unknown_relation_stays_invalid(self):
        self.assertEqual(parsed_lines("L5: after(Rohan, Tomas)")[4]["status"], "invalid")

    def test_normalisation_off_keeps_strict_parsing(self):
        text = "L2: before(whoever has packing, Tomas)\nL3: before(packing, Tomas)  # note\nL4: at(Tomas, intake)"
        parsed = parsed_lines(text, normalise=False)
        self.assertEqual([parsed[i]["status"] for i in (1, 2, 3)], ["invalid", "invalid", "invalid"])

    def test_user_message_states_line_count(self):
        messages = main.build_messages(main.parse_item(ITEMS["B2-001"]))
        self.assertIn("There are 18 lines.", messages[1]["content"])


class ExtractionFailureTests(unittest.TestCase):
    def test_unparseable_output_is_not_answered(self):
        result, _ = answer_for(SYNTHETIC, "I cannot help with that.")
        self.assertEqual((result["outcome"], result["category"]), ("failed", "no_usable_extraction"))

    def test_failed_call_is_not_answered(self):
        result = main.run_policy("single", 1, lambda _index: None, context(SYNTHETIC))
        self.assertEqual((result["outcome"], result["category"]), ("failed", "api_error"))

    def test_partial_output_is_flagged(self):
        result, _ = answer_for(SYNTHETIC, "L1: before(Ann, Ben)")
        self.assertEqual(result["category"], "degraded")
        self.assertEqual(result["unusable_lines"], [2, 3, 4])


class BudgetTests(unittest.TestCase):
    def test_one_call_at_1x_with_required_settings(self):
        client = FakeClient("L1: none")
        record = main.process_item(client, ITEMS["B2-001"], "1x", "single")
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["model"], "ibm-granite/granite-4.2-8b")
        self.assertEqual(call["temperature"], 1.0)
        self.assertEqual(call["top_p"], 0.95)
        self.assertEqual(call["extra_headers"], {"X-Item-Id": "B2-001"})
        self.assertEqual(call["extra_body"], {"reasoning": {"enabled": False}})
        self.assertEqual(record["calls"], 1)
        self.assertEqual(record["raw_outputs"][0]["finish_reason"], "stop")

    def test_call_counts_stay_within_caps(self):
        cases = (("3x", "single", 1, 1), ("3x", "fixed_vote", 3, 3), ("10x", "fixed_vote", 10, 10),
                 ("3x", "agreement", 2, 3), ("10x", "trigger", 1, 10))
        for budget, policy, low, high in cases:
            client = FakeClient("L1: none")
            main.process_item(client, ITEMS["B2-001"], budget, policy)
            self.assertTrue(low <= len(client.calls) <= high, (budget, policy, len(client.calls)))

    def test_every_call_carries_item_header_and_settings(self):
        for budget, policy in (("3x", "fixed_vote"), ("10x", "fixed_vote"), ("10x", "trigger")):
            client = FakeClient("L1: none")
            main.process_item(client, ITEMS["B2-001"], budget, policy)
            self.assertTrue(client.calls)
            for call in client.calls:
                self.assertEqual(call["extra_headers"], {"X-Item-Id": "B2-001"})
                self.assertEqual(call["model"], "ibm-granite/granite-4.2-8b")
                self.assertEqual((call["temperature"], call["top_p"]), (1.0, 0.95))
                self.assertEqual(call["extra_body"], {"reasoning": {"enabled": False}})


class PromptTests(unittest.TestCase):
    def test_prompt_examples_do_not_copy_visible_lines(self):
        texts = "\n".join(item["text"] for item in ITEMS.values())
        for line, _ in main.CORE_EXAMPLES + main.GUIDANCE_EXAMPLES:
            self.assertNotIn(line, texts)


if __name__ == "__main__":
    unittest.main()