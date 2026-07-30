# Loom user discovery

Purpose, method, and how the pieces fit together. This folder holds the
discovery kit for talking to local-LLM operators before committing v1/v2 build
effort.

## The one rule

Run these conversations to **falsify Loom's assumptions, not to collect a feature
wishlist.** Enthusiasts are generous with feature ideas, and most are solutions
in search of a problem. The value is learning which of Loom's load-bearing bets
are wrong while it is still cheap to pivot (pre-product: skeletons + one
validated model).

## Artifacts

| File | What it is |
|---|---|
| [interview-guide.md](interview-guide.md) | The questionnaire: screener + sections, past-behavior framed, each question tagged with the hypothesis it tests (H1..H8). |
| [scoring-rubric.md](scoring-rubric.md) | How to rate a response: respondent-fit tiers, the -2..+2 coding scale with per-hypothesis anchors, rollup method, and pre-registered kill criteria. |
| [response-template.csv](response-template.csv) | One row per interview. Fill during/after each call; the columns aggregate into the rollup. |

## Method

1. **Read before you ask.** Mine existing public discussion first (r/LocalLLaMA,
   the LocalLLaMA Discord, llama.cpp / exo / vLLM / Ollama issue trackers,
   Level1Techs and homelab forums, HN). Unprompted complaints beat prompted
   answers (no interviewer bias) and calibrate who the real ICP is before you
   spend anyone's time.
2. **Talk to the ICP, not general API users.** Target: homelabbers running big
   models on consumer gear; small research groups / startups pooling
   workstations; anyone who tried Petals / exo / distributed inference;
   Apple-Silicon / AMD / multi-GPU tinkerers. The colibri and ZINC authors are
   high-value first calls.
3. **Sample.** Aim for 8 to 12 Core-ICP interviews (see fit tiers in the
   rubric). Stop early only if a critical bet is decisively killed.
4. **Format.** 30-minute 1:1 calls beat a survey; you want the "why" and you
   want to be surprised. Record or take verbatim notes. Interviewer talks <= 20%
   of the time.
5. **Score each interview** with the rubric, one CSV row each.
6. **Roll up** per hypothesis and read against the decision thresholds.

## How it rolls up to a decision

Each interview scores every hypothesis on -2..+2, weighted by respondent fit.
Average per hypothesis across interviews, then:

- **Validated:** weighted mean >= +1 with low contradiction.
- **Killed:** weighted mean <= -1.
- **Inconclusive:** in between, or high variance (needs more interviews).
- **Segment split:** bimodal responses. Not a weak signal; a sign the market has
  two segments. Note which segment supports which side.

**Pre-registered kill criteria (decide these before interviewing, to blunt
confirmation bias):** the make-or-break bets are H1 (model-fit pain is real and
acute) and H2/H3 (network reality and the capacity-not-speed trade). What each
outcome means for Loom's direction is written down in the rubric so the result,
not the enthusiasm, drives the call.
