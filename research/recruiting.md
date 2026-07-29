# Recruiting interviewees

How to find people for the discovery interviews
([interview-guide.md](interview-guide.md)) without biasing them or breaking
community norms.

## Poll or no poll? Depends on the platform

The answer turns on whether the poll is anonymous.

- **Anonymous polls (Reddit, most public forums):** do not use them to filter or
  recruit. They give anonymous clicks, not contactable people. At most they are
  **engagement bait** (a comment magnet you DM into); the comments are the
  signal, not the vote counts.
- **Non-anonymous polls (WhatsApp always; Telegram with Anonymous Voting off):**
  these DO hand you named, contactable voters, so in a group you belong to they
  are a legitimate, low-friction recruiting + coarse-screen tool. Make the poll
  options self-select by fit tier, then DM the top buckets. See the poll design
  below.

Two rules regardless of platform:

- **Keep sensitive questions out of a public poll.** Visible votes suppress or
  bias anything about spend, willingness to pay, or setup details. The poll asks
  only a low-stakes interest/pain question; the real screening (hardware, pay,
  trust) happens privately in the DM.
- **Voting is not consent to an interview.** Keep the poll a no-commitment
  self-select and say you will DM to ask. Lower friction, more votes, and the
  ask stays private.

**Filter late either way.** Recruit for volume, sort at scheduling: Core first,
Adjacent as backup, nobody rejected rudely.

A form is still useful when you want more than a tier hint before the call, or in
venues without native non-anonymous polls. Poll and form are not exclusive: poll
to surface interest, DM, optionally send the form to the keen ones.

## Post principles

1. **Do not pitch the product.** Knowing what you are building biases answers and
   trips self-promotion rules. Say you are researching, not selling.
2. **Offer reciprocity.** Promise to publish the aggregated findings back to the
   group. That is what earns a yes in technical communities.
3. **Be specific about who you want and what it costs them.** "20-30 min, voice
   or text, this week" removes friction and self-selects seriousness.
4. **Respect the venue.** Check subreddit / Discord self-promo and survey rules;
   post in the right channel; if unsure, ask a mod first.

## Post - long form (Reddit / forum)

> **Title:** Researching how people actually run big models at home - looking for
> 20 min of your time (findings shared back)
>
> I am doing a batch of short interviews with people who run LLMs locally,
> especially anyone wrestling with models too big to fit comfortably on one
> machine. I want to understand what you actually run, what breaks, and where the
> real walls are (VRAM, RAM, network, patience), from people doing it, not from
> benchmarks.
>
> Not selling anything and not collecting emails for a list. I am a developer
> trying to understand the problem space before building in it, and I will post a
> write-up of what I learn back here so the whole group gets the aggregate.
>
> Good fit if you: run local models regularly, have tried (or wanted to try)
> bigger models or multiple machines, and have opinions about what is annoying.
>
> It is a 20-30 minute chat, voice or text, whenever suits you this week or next.
> If you are up for it, fill this 60-second form and I will reach out to
> schedule: [FORM LINK]. Happy to answer anything in the comments too.

## Post - short form (Discord / chat)

> Doing quick research chats with folks who run LLMs locally, especially anyone
> fighting to fit big models on limited hardware. Want to hear what you actually
> run and where it hurts. Not selling anything, and I will share the findings
> back here. 20-30 min, voice or text, your schedule. 60-sec sign-up if you are
> game: [FORM LINK] - or just reply and I will DM.

## Poll - non-anonymous group chat (WhatsApp / Telegram)

Options encode fit tier so the vote is a soft screener. DM the top buckets.

> **Caption:** Quick research poll (will share the findings back here). If you run
> models locally, how far do you push it? Tap what fits. I will DM a few folks
> who are up for a 20-min chat.
>
> **Poll (non-anonymous):**
> - Yes - I run big models / push my hardware to its limit  *(Core)*
> - Yes - and I have tried multi-machine / distributed  *(Core)*
> - Yes - but mostly smaller models on one box  *(Adjacent)*
> - I want to, but hardware holds me back  *(Adjacent, high pain)*
> - Mostly hosted APIs / cloud  *(Off-target)*

Telegram: turn **Anonymous Voting off** so you see voters; Multiple Answers is
fine. WhatsApp: non-anonymous by default, allows multi-select. Get an admin's OK
if it is not your group, and lead with the reciprocity so it does not read as
harvesting members.

## Poll - anonymous public forum (Reddit): bait only

Use only to draw storytellers into the comments, then DM them. Vote counts are
weak signal; the comments are the point.

> **What most stops you running the model you actually want, locally?**
> - Not enough VRAM / RAM to fit it
> - It fits but is too slow
> - Multi-machine setups are too fragile
> - Quantization hurts quality too much
> - Nothing, I run what I want
>
> (Comment with your setup - I am researching this and will share findings back.)

## Screener form (the real filter) - keep it under 60 seconds

Map each answer to a fit tier. Do not reject anyone here; just prioritize.

1. How often do you run LLMs locally? *(daily / weekly / occasionally / rarely)*
2. Roughly what is your biggest machine? *(GPU + VRAM, or CPU + RAM, or "not
   sure")* - free text
3. What is the largest model you have run, and at what quant? - free text
4. Have you tried running a model across more than one machine? *(yes, still do /
   yes, gave up / no, wanted to / no)*
5. What do you mainly use local models for? *(tinkering / real work / serving an
   app or other people / other)*
6. Contact for a 20-30 min chat + your timezone. - free text (Discord handle,
   email, whatever they prefer)

**Tiering from the form:** Core = frequent + (large models or multi-box or
serving). Adjacent = frequent but small/single/casual. Off-target = rare / mostly
hosted APIs.

## Logistics

- **Target funnel:** to land 8-12 Core interviews, expect to recruit 20-30
  sign-ups. Post in 2-3 venues, not one.
- **Incentive:** the shared write-up is usually enough for this crowd. A small
  gift card can raise the yes-rate but also attracts low-quality sign-ups; skip
  it unless recruiting stalls.
- **DM flow:** thank them, confirm tier fit, send 2-3 concrete time slots, keep
  it to one message. No-shows happen; over-recruit by ~30%.
- **Deliver the reciprocity.** Actually publish the findings back. It compounds:
  the write-up recruits the next round for you.
