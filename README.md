# Constraint Extraction and Solving

This project tackles the Auric AI rota benchmark using `ibm-granite/granite-4.2-8b`, the model specified in the assignment. The central design decision is to use the model for translating natural-language shift notes into formal constraints, then leave the reasoning about possible schedules to deterministic code.

The distinction matters. A language model can misread the direction of an ordering rule or mistake “between” for “immediately between”. Rather than asking it to produce a final answer directly, the pipeline checks what it extracts and evaluates the resulting constraints against every legal assignment.

## How it works

The first line of each item supplies the people, blocks, stations and station holders. The program validates this header and generates the legal assignments from the benchmark's background rules. For the standard five-person, three-station items, the search space contains 720 assignments.

The remaining lines go to the model in a single extraction prompt. Each response maps the numbered notes to a small relation language: `at`, `not_at`, `station`, `not_station`, `before`, `later`, `next` and `between`. The model is not given the answer key, and the implementation does not contain a hand-written parser for the natural-language clues.

Python validates the model's formal output against the names and values available in the current item. A line that cannot be parsed or whose relations are impossible on their own is excluded rather than silently treated as evidence. The remaining constraints are converted into bitmasks over the legal assignments. Intersecting those masks gives the solution set:

- One surviving assignment means `unique`.
- More than one means `ambiguous`, and every surviving assignment is returned.
- No surviving assignments means `inconsistent`.

For inconsistent items, the program cites an inclusion-minimal unsatisfiable subset of note lines. The brief states that every valid conflict set on the visible benchmark has at least three statements, so a two-line contradiction signals a misread line: the program prefers a minimal set of three or more lines whenever one exists, and otherwise cites the smallest set and flags it as `short_conflict` in the status file. It verifies that the selected subset is inconsistent and that removing any one of its lines makes it satisfiable. The output cites the original note text, not a reconstructed paraphrase.

## Working within the call budget

The entry point is `run`. The `1x` budget makes exactly one model call per item. Under `3x` and `10x`, the pipeline can collect additional extractions when the current voted result still contains unusable lines or leaves more than four possible assignments. A contradictory result (no possible assignment) counts as settled and is not re-sampled, and a tie between samples keeps the earlier reading. The same prompt is used for every call; later calls do not depend on the wording of earlier responses.

For each line, the aggregation step selects the interpretation supported by the most samples, with unusable outputs abstaining. The solver then evaluates the combined constraints. Calls are counted per item, client retries are disabled, and the program enforces the specified budget.

Every request uses `temperature=1.0`, `top_p=0.95`, `max_tokens=2000`, reasoning disabled through `extra_body={"reasoning": {"enabled": False}}`, and an `X-Item-Id` header containing the item's id. Each item receives exactly one call at `1x`, between one and three at `3x`, and between one and ten at `10x`. A failed request still counts as a call, and a per-item counter stops the run if any item would exceed its budget.

## Experiments

I built `ablation.py` to separate the cost of collecting model outputs from the cost of evaluating them. It records samples once and replays them through the pipeline offline, so competing conditions can be compared without making new model calls for each condition.

The experiments compare the multi-call policies, line-level and answer-level voting, conflict-set methods, parser normalisation, distractor guidance and a direct-answer baseline without the solver. The policy-selection rule was fixed before measurement: a higher-call policy would replace a cheaper one only if it scored better on the first 30 items in every recorded run.

The final runs and ablation summary are in `results/`. The accompanying report gives the measured scores, comparisons and limitations. The results should be read as evidence from the supplied visible benchmark, not as a held-out performance estimate. The answer and status files of the three live runs were regenerated offline from their recorded model replies when the conflict-citation rule changed; the model calls and replies themselves are unchanged. A variant that also re-samples contradictory results at 3x and 10x (`trigger_conflict`) is evaluated in the ablation but not shipped.

## Running the pipeline

Use Python 3.12.10. Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Keep this environment activated when running `./run`; the launcher uses the `python3` of the active environment. Create a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=ibm-granite/granite-4.2-8b
```

The `.env` file is git-ignored and must not be committed. Environment variables take precedence over values in `.env`. The program stops if `MODEL` is set to a different model. `AURIC_WORKERS` can optionally control the number of items processed in parallel; its default is four.

Run an item set with one of the supported budgets:

```bash
./run items.json --budget 1x --out answers.json
```

Replace `1x` with `3x` or `10x` to use the corresponding call policy. The answers are written to `answers.json`; per-item diagnostics are written separately to `answers.json.status.json`.

Exit code `0` means every item was answered, `3` means the output was written but at least one item could not be answered, and `2` indicates a usage, configuration or input error.

To score an output against a key, run:

```bash
python3 score.py <key.json> answers.json
```

## Tests and reproducibility

The unit tests use a fake model client and do not make API calls:

```bash
python3 -m unittest -v test_main
```

They cover relation semantics, answer classification, conflict minimality, invalid and contradictory lines, parser normalisation, model settings, call limits and failure handling. They also check that the answer changes when the model output changes.

To collect ablation samples, run the required collection commands for each arm and repeat:

```bash
python3 ablation.py collect --arm E --run 1
python3 ablation.py collect --arm E --run 2
python3 ablation.py collect --arm N --run 1
python3 ablation.py collect --arm D --run 1
python3 ablation.py evaluate --key <key.json>
```

Arm `E` uses the shipped extraction prompt, `N` removes the distractor guidance, and `D` asks the model to answer directly without the solver. Collection makes ten calls per item and can resume after interruption. Evaluation reuses the recorded samples and makes no API calls.

The recorded samples behind the reported ablation table (`results/samples_*.jsonl`) are included, so `evaluate` can be re-run without API calls. It scores against the visible answer key, which is not distributed with this repository; pass your own copy with `--key`. `collect` skips items already recorded under the same arm and run number, and `evaluate` uses every recorded run in `results/`. To re-collect from scratch, first move the existing `results/samples_*.jsonl` files out of `results/`, or use new run numbers.

## Repository contents

- `main.py` contains the extraction, validation, solving and aggregation pipeline.
- `ablation.py` records model outputs and evaluates the ablation conditions.
- `test_main.py` contains the offline tests.
- `run` is the command-line entry point.
- `results/` contains the answers and per-call status files of the reported live runs (`run3_1x`, `run4_3x`, `run5_10x`), the recorded ablation samples, and the ablation summary.
- `items.json`, `example_answers.json` and `score.py` are supplied assignment files; the assignment brief is not redistributed.

The API key and `visible_key.json` are intentionally excluded from version control.

The two-page report, submitted separately, discusses the quantitative results, the main failure modes and the limits of the evaluation.