# Redica Collaboration Meeting — Full Speaker Narrative
**Date:** June 2026  
**Duration:** 60 minutes  
**Audience:** Redica technical team (feature engineers + domain experts)

---

## Slide 1 — Title
*[1 minute]*

Good morning — thank you for making time for this. We have been looking forward to this conversation for a while, because the work you have done at Redica on 483 classification is exactly the kind of external benchmark we need to test our own approach against.

A quick framing before we start: this is not a presentation in the traditional sense. We are going to show you what we built and why, and we genuinely want you to push back where you disagree. The goal of this hour is alignment — we want to walk away with a shared understanding of where our systems agree, where they don't, and what those differences mean for research. So please interrupt, ask questions, and tell us when something looks wrong.

---

## Slide 2 — Why we care about 483 observations
*[5 minutes]*

Let me start with the research question that motivates everything we are going to show you.

We are studying drug shortages — specifically, why generic drugs go into shortage and whether we can predict it before it happens. This matters because shortages of generic drugs are not random. They cluster around specific manufacturers, specific facilities, and specific failure modes. And when a facility fails, it often fails in ways that are visible — to the FDA, and therefore to us — well before a shortage actually materializes.

The signal we are most interested in is the FDA Form 483. When an FDA investigator completes an inspection and finds a violation, they issue a 483 observation. It is the most granular public record of a quality problem at a specific manufacturing facility. Not a summary, not a conclusion — the actual documented finding, in the inspector's words.

Our hypothesis is that patterns in those findings — the types of violations, how severe they are, how broadly they affect the facility, whether the facility responds — can predict downstream outcomes: Class I and II drug recalls, adverse event signals in FAERS, and confirmed drug shortages.

The challenge is that 483 observations are free text. Every investigator writes differently. The same underlying violation can be documented in a dozen different ways. To use these observations as model features, we need to convert them into structured signals that a machine learning model can learn from. That is what we built.

---

## Slide 3 — Our data universe
*[6 minutes]*

Before we get into the classification system, let me show you where we stand on data coverage, because this is also part of what we want your help with.

We are studying 127 manufacturing facilities — these are the generic drug manufacturers that produce the 14 APIs we focus on for our shortage prediction model. Across those 127 facilities, Redica's own event log shows approximately 853 total inspections that resulted in a 483, going back to 1998.

Of those 853, roughly 549 are pre-2018. You did not share documents for those, which we completely understand — that is a large backfill. That leaves about 280 post-2018 inspections. Of those, you shared 246 actual 483 documents, which is 88 percent coverage of the post-2018 period. That is very good. We processed those 246 documents through our pipeline and extracted 1,083 individual observations.

The gap we want to flag is the remaining 34 post-2018 inspections that appear in your event log but where no documents were included in the data share. We can see in your audit trail that those inspections happened. We just do not have the documents. Later in the meeting we will ask whether there is a reason those were excluded, and whether we can get them.

We also independently downloaded 82 PDFs from the FDA's public dashboard, covering 38 facilities. Thirty of those are pre-2018 — that is unique historical data we would not have otherwise. But the text quality from OCR is noticeably lower than what you shared with us. Your text is cleaner, more complete, and structured much better for downstream processing. So for the primary run we are presenting today, we used your data exclusively.

Of the 127 FEIs in our study, 29 have no observation data at all. Five never received a 483 — they were always inspected clean. Twelve only have pre-2018 483s. And twelve have post-2018 483s in your event log with no documents. That last group is the most actionable, and we will come back to it.

---

## Slide 4 — How we extract features: the LLM pipeline
*[6 minutes]*

Here is what our extraction pipeline looks like at a high level.

We take each observation text verbatim — the raw text from the 483 document, exactly as the investigator wrote it — and we pass it to Anthropic's Haiku 4.5 language model with a structured prompt. The prompt is engineered to output a strict JSON object with ten defined fields. There is no free generation — the model is constrained to produce exactly the fields we define, in exactly the format we specify.

The key design principle is that every field is independently defined. We do not ask the model to summarize or interpret — we give it a specific question for each dimension with anchor examples that show what a correct answer looks like for that dimension. The model reads the text once and answers all ten questions simultaneously.

Let me show you a concrete example, because this is the best way to understand what the system does.

The observation text is: *"Batch production records were not completed at the time of manufacture. Entries were reconstructed from memory 2 days after processing."*

For severity, the model assigns Major. The rule it is following says: Major means the text documents an actual defect or confirmed failure at the facility. Reconstructed batch records is a confirmed failure — it happened. So it is not Moderate, which is reserved for systems that could fail.

For data integrity, the model marks True. The rule says: mark true only for explicit data trustworthiness failures — falsification, backdating, reconstructed records. This observation has a textbook example of that.

For scope, the model assigns SingleBatch — the text says "batch production records," implying a single event.

For root cause, the model assigns Cultural — this is a behavioral failure, not a missing piece of equipment.

For remediation signal, the model assigns None — there is no corrective action language in the text.

And the investigation flag is False — the text does not say an investigation was missing; it documents what happened.

This is what we mean by structured extraction. Six of the ten dimensions, each independently reasoned from the same text, each following a precise rule.

---

## Slide 5 — Our classification system in full detail
*[6 minutes]*

Here is the full system — all ten dimensions in one view.

The first group is categorical: violation category, which assigns each observation to one of eight CFR Part 211 domains; severity tier, which is four levels; scope of failure, which is four levels; root cause type, also four levels; and remediation signal, which is an ordinal with four levels from Strong to None.

The second group is binary flags — each is either True or False. Data integrity flag, repeat flag for cross-inspection recurrence, patient risk flag for direct patient safety implications, contamination flag for physical or microbial contamination, and investigation flag for failure to investigate a documented deviation.

And then two regex-based flags that we derive programmatically from the text without LLM involvement: the OOS/OOT flag fires whenever out-of-specification or out-of-trend language appears, and the warning letter reference flag fires when the text references a prior warning letter.

I want to pause here and ask you directly: which of these dimensions make sense to you from your experience classifying 483 observations? Which have you seen attempted before in the industry? And which ones make you skeptical — either because the concept is hard to operationalize from text, or because you think the definition is not quite right? We will circle back to the unique-to-us dimensions later, but if something jumps out now, please say so.

---

## Slide 6 — Redica's classification system in full detail
*[7 minutes]*

Now let me show how we understand your system, based on the data you shared with us and the documentation we have reviewed. We want you to correct anything we have wrong here.

On severity: you use a three-tier system — Critical, Major, and Other — anchored to PIC/S GMP deficiency classification. The Critical tier is for deficiencies likely causing direct risk to patient health. Major is significant non-compliance that may cause a product defect, but not necessarily a confirmed one. Other covers anything else that departs from GMP but does not reach Major. At the document level, you roll up: a single Critical observation makes the document Critical; one to five Majors makes it Major; all Others makes it Minor.

On domain: you use six QSL areas. Quality Unit is the largest at 36 percent of observations in our dataset, followed by Production and Laboratory at around 19 and 18 percent each, then Facilities and Equipment also at 18 percent, Materials at 3 percent, and Packaging and Labeling at 1 percent. Each area also has Level 1 sub-labels — for example, Production breaks down into Sterile Products, Process Control, Contamination Control, Training, and so on.

On data integrity: this is where your system is richest. Of the 1,083 observations in our dataset, you flagged 143 — about 12.8 percent — as having a data integrity issue. Within those 143, you use a 13-label taxonomy. The top four labels by frequency are System Controls at 26 percent of DI observations, Contemporaneous at 21 percent, Complete at 15 percent, and Attributable at 12 percent.

We have mapped these against the ALCOA+ framework — Attributable, Legible, Contemporaneous, Original, Accurate, plus Complete, Consistent, Enduring, Available. Several of your labels map cleanly: Contemporaneous is the C in ALCOA, Complete and Attributable are direct ALCOA+ principles, Data Destruction maps to Enduring, Backup and Archival maps to Enduring and Available, Original Data maps to the O.

What is less clear to us is System Controls at the top of your list. That looks more like 21 CFR Part 11 — electronic records and electronic signatures — than a pure ALCOA principle. And Testing into Compliance is interesting — that is not a named ALCOA+ dimension, but it is a well-recognized FDA enforcement concept. We want to ask: are those labels ALCOA-derived, or were they developed separately from your regulatory work?

Our approach for now is a binary flag — True or False. We are holding off on sub-classification until we understand your taxonomy better, which is exactly why this conversation matters.

---

## Slide 7 — How the two systems compare
*[8 minutes]*

Now let me put the two systems side by side.

On severity and domain, we are directly comparable. For severity, we can collapse our Moderate and Minor tiers into your Other tier — at that level of aggregation, our distributions are nearly identical and our agreement rate is 79 percent. For domain, five of your six QSL areas map directly to our eight-class CFR taxonomy. The sixth — your Quality Unit — is where the interesting disagreement lives, and we will come back to it on the next slide.

On data integrity, we are currently using a binary flag and you have 13 labels. At the binary level — flagged or not — our rates are close: you flag 12.8 percent, we flag approximately 17 percent. The question for today is whether your 13-label taxonomy was purpose-built for a downstream model, or whether it emerged from a different need. Because if it feeds a risk score or a predictive model, we want to adopt a compatible approach. If it is primarily for regulatory audit use, we might stay binary and just cross-validate against your flags.

The dimensions unique to us are where we most need your help. Scope, root cause, remediation signal, and four of our binary flags — repeat inspection, patient risk, contamination, and investigation failure — have no equivalent in your system. We are not asking you to validate the data today. We are asking you, after the meeting, to read through our prompt definitions and tell us whether they are conceptually sound from a GMP standpoint.

What you have that we do not: your AI-generated observation summaries are cleaner than our OCR text, and your QSL Level 1 sub-labels are more granular than our eight-class taxonomy. Both of those could directly improve our pipeline.

One question I want to put to you now: when you see an observation about inadequate training, or about incomplete recordkeeping, where does that go in your QSL taxonomy? In our system those go to OrgPersonnel and RecordsReports respectively. In your data, they both appear to land in Quality Unit. Is that intentional? Is it that the quality unit is responsible for training and records — so you assign to the responsible body rather than the subject matter of the violation?

---

## Slide 8 — Where we agree and where we still differ
*[6 minutes]*

Let me give you the numbers.

On severity: we are effectively calibrated. Redica calls 70 percent of observations Major or Critical. We call 67 percent — a three-percentage-point gap. Earlier versions of our system, built on a different LLM, had a 25-point gap because our prompt was too conservative — it required confirmed patient harm before calling something Major. We rewrote the severity definition to align with PIC/S and the gap closed dramatically. The 79 percent agreement rate across 1,066 matched observations is, in our view, strong enough that severity is no longer an open question between our two systems. We consider this resolved.

Domain is the persistent disagreement. We agree 65 percent of the time, which is up from 62 percent with the prior prompt. The most common mismatch is 83 cases where you assigned QualitySystem and we assigned ProductionControls, and 79 cases where you assigned QualitySystem and we assigned LabControls.

This disagreement is systematic and interpretable. It is not noise — it reflects a genuine difference in classification philosophy. We assign to the technical domain where the failure physically occurred: if a lab test was not performed correctly, that is a lab controls issue. You appear to assign to Quality Unit when the quality unit was responsible for overseeing that domain. Both are defensible.

The question we most want to answer in this meeting is: which philosophy better predicts downstream outcomes? If a lab failure matters because it reflects a quality unit oversight breakdown, then your assignment might carry more predictive signal. If it matters because the specific lab process failed, then ours might. We do not know yet — we have not run the outcome models. But understanding your reasoning will help us decide whether to realign our prompt or keep the domain split as a feature.

---

## Slide 9 — What we are asking from Redica
*[5 minutes]*

Three concrete asks.

First, data. We have 29 facilities in our study with no 483 observation data at all. Twelve of those have confirmed post-2018 inspections in your event log but no documents were shared with us. Can you tell us why those were excluded? Is it a licensing issue, a document quality issue, something else? Even partial coverage of those 12 would materially improve our model. And if any pre-2018 documents exist, even a sample, that would also be valuable — that is our only historical window into early-period quality signals.

Second, guidance on classification. On severity, we want to understand the practical annotator boundary between Major and Other. Does your annotator need to see evidence of a confirmed defect to call something Major, or does significant non-compliance alone qualify? That distinction drove our 25-point severity gap with the prior prompt, and getting it right matters. On domain, the training and documentation question I raised — we want your explicit answer on where those sit in your QSL taxonomy, and what the reasoning is.

Third, expert validation of our unique dimensions. Scope, root cause type, remediation signal, and the binary flags are novel — we developed them specifically for this research. There is no published benchmark, no industry standard. We are going to share the full prompt document with you after this meeting. What we are asking is that someone on your team reads through the definitions and tells us: do these make regulatory sense? Are the boundaries reasonable from a GMP standpoint? Would an experienced investigator recognize what we are trying to capture? That validation is required before we can publish these as research features.

---

## Slide 10 — Open questions for today
*[5 minutes + discussion]*

Let me close with the three questions we most want to walk away having answered.

On severity: when your annotators call something Major rather than Other, what is the test they are applying? Is it "this could cause a product defect" — the non-compliance standard — or "this did cause a defect" — the confirmed-failure standard? Our current prompt uses the confirmed-failure standard, which gets us to 79 percent agreement. We want to know if that is the right philosophy for your system or whether we are getting there by accident.

On domain: the Quality Unit question. Is domain assignment in your system about where the failure happened, or about who was responsible for preventing it? We genuinely do not know the answer and it is consequential — it determines whether a 35 percent domain disagreement is a bug we should fix or a feature worth keeping.

On data integrity: this is the biggest open question. You built a 13-label ALCOA-based taxonomy. We are currently using a binary flag. We want to understand what drove that decision. Was it that your downstream risk model needed the sub-type granularity? Was it that your regulatory clients needed ALCOA-level traceability? Was it that the ALCOA framework itself suggested the structure? The answer tells us whether we should adopt a similar taxonomy — which we are prepared to do — or whether binary is adequate for predictive research.

What happens next: we will share the full prompt rules document after this meeting. We will update our DI approach to match your ALCOA taxonomy based on what we learn today. And the goal, ultimately, is a shared classification framework that we can both stand behind when we publish — one that has both the regulatory grounding of your system and the research operationalization we need for our model.

Thank you. Let us open it up.

---

*Total estimated runtime: 56 minutes presentation + 4 minutes buffer for questions mid-slides.*
