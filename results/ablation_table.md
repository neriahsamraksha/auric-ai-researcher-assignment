# Ablation results

Recorded samples: arm E: 2 run(s), arm N: 1 run(s), arm D: 1 run(s). Prompt versions: E=8a2b92491001, N=028640b01747, D=87b6dde1abdb.
Budget k replays the first k recorded samples of each item. Each cell: mean macro exact match on all items [value per run], mean model calls per item.
Shipped policy: 3x = trigger, 10x = trigger. Rule selects: 3x = trigger, 10x = trigger (a policy using more calls replaces a cheaper one only if its dev-half macro score is higher in every recorded run; dev half = first 30 items of items.json; selection scored with the smallest-first conflict rule in force when it was fixed).

| Condition | 1x | 3x | 10x |
|---|---|---|---|
| Full system (shipped policy) | 0.625 [0.550, 0.700], 1.0 calls | 0.658 [0.583, 0.733], 1.5 calls | 0.683 [0.600, 0.767], 2.0 calls |
| Multi-call off: one call | 0.625 [0.550, 0.700], 1.0 calls | 0.625 [0.550, 0.700], 1.0 calls | 0.625 [0.550, 0.700], 1.0 calls |
| Line vote over all calls | 0.625 [0.550, 0.700], 1.0 calls | 0.675 [0.633, 0.717], 3.0 calls | 0.717 [0.700, 0.733], 10.0 calls |
| Answer vote over all calls | 0.625 [0.550, 0.700], 1.0 calls | 0.650 [0.583, 0.717], 3.0 calls | 0.667 [0.650, 0.683], 10.0 calls |
| Stop when two answers agree | 0.625 [0.550, 0.700], 1.0 calls | 0.667 [0.617, 0.717], 2.4 calls | 0.658 [0.617, 0.700], 2.5 calls |
| Stop when extraction is clean | 0.625 [0.550, 0.700], 1.0 calls | 0.658 [0.583, 0.733], 1.5 calls | 0.683 [0.600, 0.767], 2.0 calls |
| Also re-sample contradictions (not shipped) | 0.625 [0.550, 0.700], 1.0 calls | 0.675 [0.600, 0.750], 2.2 calls | 0.733 [0.650, 0.817], 5.4 calls |
| Conflict set: smallest first (previous rule) | 0.583 [0.517, 0.650], 1.0 calls | 0.617 [0.550, 0.683], 1.5 calls | 0.642 [0.567, 0.717], 2.0 calls |
| Conflict set by deletion order | 0.617 [0.550, 0.683], 1.0 calls | 0.650 [0.583, 0.717], 1.5 calls | 0.675 [0.600, 0.750], 2.0 calls |
| Conflict minimisation off | 0.375 [0.317, 0.433], 1.0 calls | 0.408 [0.350, 0.467], 1.5 calls | 0.425 [0.367, 0.483], 2.0 calls |
| Parser normalisation off | 0.617 [0.550, 0.683], 1.0 calls | 0.650 [0.583, 0.717], 1.5 calls | 0.675 [0.600, 0.750], 2.1 calls |
| Distractor guidance removed (arm N) | 0.533 [0.533], 1.0 calls | 0.550 [0.550], 1.8 calls | 0.550 [0.550], 3.3 calls |
| Solver removed: direct answer, answer vote (arm D) | 0.000 [0.000], 1.0 calls | 0.000 [0.000], 3.0 calls | 0.000 [0.000], 10.0 calls |

| Budget | Policy | Mean calls | Dev-half macro per run | All-items macro (mean) |
|---|---|---|---|---|
| 3x | single | 1.00 | 0.492, 0.686 | 0.583 |
| 3x | fixed_vote | 3.00 | 0.583, 0.686 | 0.625 |
| 3x | agreement | 2.34 | 0.553, 0.686 | 0.617 |
| 3x | trigger | 1.47 | 0.523, 0.746 | 0.617 |
| 3x | answer_vote | 3.00 | 0.492, 0.686 | 0.600 |
| 10x | single | 1.00 | 0.492, 0.686 | 0.583 |
| 10x | fixed_vote | 10.00 | 0.655, 0.758 | 0.667 |
| 10x | agreement | 2.52 | 0.553, 0.655 | 0.608 |
| 10x | trigger | 1.96 | 0.564, 0.849 | 0.642 |
| 10x | answer_vote | 10.00 | 0.625, 0.655 | 0.617 |
