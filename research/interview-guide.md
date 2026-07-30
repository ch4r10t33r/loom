# Loom discovery interview guide

A 30-minute, past-behavior interview for local-LLM operators. Each question is
tagged with the hypothesis it feeds (H1..H8, defined in
[scoring-rubric.md](scoring-rubric.md)). You will not ask every follow-up; follow
the energy and dig where there is signal.

## Interviewer ground rules

- **Do not pitch Loom.** If they know what you are building, they will tell you
  what you want to hear. Describe it only in the wrap-up, if at all.
- **Ask about last week, not hypotheticals.** "What did you run yesterday" beats
  "would you use X." The one hypothetical is saved for the wrap.
- **Follow "why" three times.** The first answer is rarely the real reason.
- **Shut up.** Silence pulls out the real answer. Target <= 20% of the talking.
- **Capture verbatim.** Exact quotes are worth more than your paraphrase; write
  down surprises and anything not on this sheet.

## Section 0 - Screener (decides respondent fit)

1. Do you run LLMs locally? How often, and for how long have you been at it?
2. Walk me through your setup: machines, GPUs/CPU, RAM, disk. *(probe: one box or
   several? how are they networked?)*
3. What did you actually run in the last week or two? *(probe: models, sizes,
   quant)*
4. What do you use local models *for*? *(probe: hobby/tinkering, real work,
   serving other people/apps)*
5. Have you ever tried running a model across more than one machine?

> Use answers to 1-5 to set the fit tier (Core / Adjacent / Off-target) per the
> rubric. Keep going regardless, but weight the signal accordingly.

## Section 1 - Hardware and network reality  [H2]

6. What is the fastest and slowest machine you run models on?
7. How are your machines connected to each other? Wired or wifi, and do you know
   the link speed? *(probe: 1GbE / 2.5 / 10GbE / Thunderbolt / none)*
8. Is your model storage on NVMe, SATA SSD, or spinning disk?
9. Have you ever hit a wall where the network or disk was the bottleneck, not the
   GPU/CPU? Tell me about it.

## Section 2 - Models and the fit problem  [H1, H7, plus model-target data]

10. Is there a model you *wanted* to run but could not fit on your hardware? What
    happened next? *(probe: gave up? smaller model? cloud? bought hardware?)*
11. When a model does not fit, how much does that actually bother you on a scale
    you would describe in your own words? *(listen for intensity, not a number)*
12. What quantization do you usually run, and where does quality get too degraded
    for you? *(probe: is int4 acceptable for your use?)*
13. Which 2-3 models do you most wish you could run well locally today?

## Section 3 - Workload shape and latency  [H4]

14. When you use a local model, is it one-at-a-time interactive chat, or are you
    batching / serving an app / running agents / serving other people?
15. How much does per-token speed matter to you versus just getting the job done?
16. Would a setup that runs a much bigger model but slower per token be useful to
    you, or useless? *(probe: for which of your use cases?)*  [H3]

## Section 4 - Multi-machine and distributed history  [H5]

17. (If they tried distributed) What did you use, and what made you stop or keep
    going? *(probe: Petals, exo, llama.cpp RPC, vLLM, Ray)*
18. (If they did not) What has kept you from pooling machines?
19. Would you pool with machines you do not own - a friend's, a colleague's, a
    stranger's? Where is your line? *(probe: same household / same team /
    trusted club / open internet)*

## Section 5 - Trust, sharing, and paying  [H6]

20. Would you let your machine serve model weights or compute to other people if
    it was idle? What would you want in return?
21. Would you pay to use spare capacity on other people's machines to run a
    model you cannot fit? *(probe: how much, versus renting a cloud GPU?)*
22. What would make you *not* trust a shared/pooled setup? *(listen for:
    poisoned weights, privacy, someone seeing my prompts, reliability)*

## Section 6 - Operational pain and abandonment  [H8]

23. Think of the last local-LLM tool you stopped using. Why did you drop it?
24. What is the most annoying part of your current setup - install, model
    conversion/quant, updates, OOM, config?
25. How do you handle model updates or trying a new version today?

## Section 7 - Why local at all  (value-prop data)

26. You could just use a hosted API or rent a GPU. What keeps you running local?
    *(probe: privacy, cost, offline, data control, tinkering, latency,
    principle)*
27. If your local setup vanished tomorrow, what would you lose that you could not
    easily replace?

## Section 8 - Wrap  (the one allowed hypothetical + recruiting)

28. Magic wand: one thing about running models locally is suddenly solved. What
    did you pick? *(this is a wishlist question - treat as color, not proof)*
29. Who else should I talk to who runs this stuff seriously?
30. Anything I did not ask about that I should have?

## After the call (within 15 minutes, while fresh)

- Fill one row in [response-template.csv](response-template.csv).
- Score H1..H8 per the rubric anchors.
- Record: fit tier, the 2-3 best verbatim quotes, top model targets they named,
  their hardware/network/backend, and any surprise or new pain not on this sheet.
