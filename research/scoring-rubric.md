# Loom discovery scoring rubric

How to turn interview answers into a signal you can act on. Score every interview
the same way, capture one row per interview in
[response-template.csv](response-template.csv), then roll up per hypothesis.

## Step 1 - Respondent fit tier (a weight, not a score)

Only weight signal by how much the person is a real target user. From the
screener (Section 0):

| Tier | Definition | Weight |
|---|---|---|
| **Core** | Runs local LLMs regularly AND (wants/runs large models OR has a multi-box / serving / agent workload) | 1.0 |
| **Adjacent** | Runs local LLMs regularly but only small models, single box, casual | 0.5 |
| **Off-target** | Mostly uses hosted APIs; local is incidental | 0.0 (record, do not count) |

Report every rollup twice: Core-only, and Core+Adjacent (weighted). If they
disagree, say so.

## Step 2 - The hypotheses (what each bet is)

| ID | Loom is betting that... | Tested by |
|---|---|---|
| **H1** | The "big model will not fit" pain is real and acute for the ICP | Q10, Q11, Q13 |
| **H2** | Deployments have network fabric fast enough for the speed story (else it is capacity-only) | Q6, Q7, Q9 |
| **H3** | "Runs a bigger model but slower per token" is an acceptable trade | Q15, Q16 |
| **H4** | The workload is serving / batch / multi-user, not purely single-stream snappy chat | Q14, Q15 |
| **H5** | Users will pool across multiple machines (including ones they do not own) | Q17, Q18, Q19 |
| **H6** | There is real demand for the compensation / economy layer | Q20, Q21 |
| **H7** | int4-class quantization quality is acceptable for their use | Q12 |
| **H8** | Ease-of-use / ops is a decisive adoption factor Loom can win on | Q23, Q24 |

Also **capture (not scored, just tallied):** model targets named (Q13), hardware
and GPU vendor (Q2), why-local reasons (Q26), and any new pain (Q30).

## Step 3 - Code each hypothesis on -2..+2

Use the anchors. If a hypothesis was not really covered in the call, mark `NA`
(blank), do not guess.

| Score | Meaning |
|---|---|
| **+2** | Strong, unprompted support. They live this problem / behavior. |
| **+1** | Leans support, with qualifiers. |
| **0** | Genuinely mixed or neutral. |
| **-1** | Leans against. |
| **-2** | Strong contradiction. Their behavior directly refutes the bet. |

### Per-hypothesis anchors (what +2 and -2 look like)

- **H1** +2: named a specific model they gave up on and it clearly frustrates
  them. -2: "I just use a smaller model, never think about it."
- **H2** +2: has 10GbE / Thunderbolt / fast interconnect and uses it. -2: wifi or
  1GbE only, and has felt it as a wall.
- **H3** +2: "I would happily wait if it meant running the big one." -2: "slow
  per token is a non-starter for everything I do."
- **H4** +2: batches, serves an app, runs agents, or serves other people. -2:
  purely one-at-a-time interactive chat, latency-sensitive.
- **H5** +2: already pools machines, or eager to (incl. others'). -2: "only ever
  my own single box, no interest in pooling."
- **H6** +2: would pay for capacity or expects to earn by serving, with a number.
  -2: "I would never pay for or trust someone else's machine."
- **H7** +2: runs int4/Q4 happily. -2: insists on Q8/FP16, int4 quality
  unacceptable.
- **H8** +2: abandoned tools over setup/ops friction and would switch for less.
  -2: current tooling is fine, ops is not a factor.

## Step 4 - Roll up

For each hypothesis: weighted mean = sum(score x fit_weight) / sum(fit_weight)
over interviews that scored it (skip NA).

| Weighted mean | Verdict |
|---|---|
| >= +1.0 | **Validated** |
| -1.0 to +1.0 | **Inconclusive** (or segment split - see below) |
| <= -1.0 | **Killed** |

- **Watch variance.** A mean near 0 from tight scores is genuine ambivalence. A
  mean near 0 from a pile of +2s and -2s is a **segment split** - two markets.
  Note which respondent type sits on each side; that is a positioning finding,
  not noise.
- **Minimum n.** Do not call a hypothesis Validated or Killed on fewer than ~6
  Core interviews. Below that, everything is Inconclusive.

## Step 5 - Pre-registered kill criteria (decide the consequence now)

Written before interviewing so the result drives the decision, not the
enthusiasm:

- **If H1 is Killed** (fit pain is not real): Loom's core premise is invalid for
  this ICP. Stop and reconsider the target user or the product before building
  v1.
- **If H2 is Killed but H3 is Validated:** reposition Loom as a **capacity-only**
  "make it fit" tool, drop the speed claims from messaging, and deprioritize the
  fast-fabric transport work. Still viable.
- **If H2 and H3 are both Killed:** the distributed value prop is in serious
  question; single-box streaming (pinned colibri-style) may serve this ICP
  better than the swarm. Escalate to a design rethink.
- **If H5 or H6 is Killed:** keep v1 as **trusted, own-machines-only pooling**;
  shelve the untrusted-peer economy (v2) until a different ICP appears that wants
  it. This de-risks a large speculative bet.
- **If H4 is Killed** (everyone wants snappy single chat): Loom's serving-first
  latency story is a mismatch; either narrow to the batch/serving niche that
  does exist or rethink the latency-hiding plan.

## Bias guardrails

- Score from what they *did*, not what they *said they would do*. Discount Q28
  (magic wand) and any hypothetical.
- One person's vivid story is an anecdote, not a trend. Wait for the rollup.
- If you find yourself scoring +2 because you liked the person, re-read the
  anchor.
