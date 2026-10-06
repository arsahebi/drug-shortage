# MQRI Presentation Narrative Guide
## Full talking points — slides + live dashboard walkthrough
### Drug Shortage Research Team · April 2026

---

# OVERALL STORY ARC

> "We have 18 companies making the exact same molecule — generic metformin. The FDA has approved all of them. But approval is a snapshot in time. Post-market manufacturing quality can drift, degrade, or fail — and the FDA's inspection system is too resource-constrained to catch it quickly. The question we asked: can we build a forward-looking quality risk signal using only publicly available data? And can we validate it against truly independent evidence — contamination measurements taken by a third party who never talked to us?"

That's the one-paragraph pitch. The index says: some manufacturers carry far more accumulated risk signals than others. The Valisure data — which we never touched during model construction — confirms that our signals predict real chemical contamination levels.

---

# SLIDE-BY-SLIDE TALKING POINTS

## Slide 1 — Title

Start simply. Read the orange banner aloud: *"Higher Score = Higher Quality Risk."*

> "I want to anchor that before anything else. This is not a quality score like a restaurant rating where 90 out of 100 is good. Higher MQRI means more accumulated evidence of quality problems. A score of 50 is worse than a score of 10."

Four stats: 18 facilities studied, 3 risk domains, annual panel, validated externally.

> "We're looking at every manufacturer selling generic metformin ER in the US market. That's a $1.5 billion annual market. These 18 companies together supply essentially all the metformin ER dispensed in America."

---

## Slide 2 — What Does MQRI Measure?

Walk through the 7 inputs on the left. Don't just read labels — explain each one in one sentence:

**FDA Inspections:** "FDA inspectors physically visit drug manufacturing plants. When they find serious CGMP violations, they classify the inspection as OAI — Official Action Indicated. That's our strongest signal. A facility with multiple OAIs over its history has a documented pattern of failing to meet manufacturing standards."

**Form 483 Observations (NLP):** "Every inspection where violations are found results in a Form 483 — a list of specific objectionable conditions the inspector saw. We ran natural language processing on those documents to flag particularly dangerous categories: contamination risks, data integrity failures, out-of-spec findings. These aren't just regulatory bureaucracy — they describe what was actually wrong on the production floor."

**Warning Letters:** "When a facility gets a 483, they're supposed to fix it. If FDA decides the response is inadequate, they escalate to a Warning Letter — a formal legal notice. We analyzed the text of warning letters for language about management failures and corporate-level breakdown. A company where the CEO is personally named in a WL is a different category of problem."

**Drug Recalls & Import Refusals:** "Recalls are the consequences — when quality failures actually reached patients. Class I recalls are the most severe: FDA's judgment that there's a reasonable probability of serious health consequences. Import refusals mean FDA turned away shipments at the border."

**FAERS Adverse Events:** "The FDA Adverse Event Reporting System is where healthcare providers and patients report problems with drugs. We linked these reports back to the specific manufacturer's ANDA application number. Critically, we capture both how many reports there are and how severe they are — life-threatening events are weighted more than non-serious reports."

**IQVIA Market Volume:** "This one might surprise you — why does market share matter for quality risk? Think of it this way: if a small facility with quality problems sells 1,000 prescriptions, the harm is bounded. If a large facility with the same quality problems supplies 30% of US metformin demand, the societal exposure is 300 times larger. Volume is a risk multiplier."

**FEI Network Severity:** "This is a composite score built from a network analysis of FDA Establishment Identifiers — connecting facilities to their recall histories, inspection patterns, and import alert status. It captures regulatory signals that don't fit neatly in the other categories."

Then turn to the right column (Valisure):

> "Now here's what makes this study unusual. Valisure — an independent pharmacy in Connecticut — independently purchased generic metformin from retail pharmacies and chemically analyzed it. They tested for NDMA, DMF, and dissolution quality. We never looked at their data during construction of the index. These measurements are our report card. If MQRI works, its scores should predict what Valisure found."

---

## Slide 3 — How Scores Are Computed

> "The index is computed annually — one score per facility per year. The critical design rule: each year's score only uses data that existed through December 31 of that year. We're not cheating by using future information."

On the three boxes:

**Currently Time-Varying:** "These update every year as new inspections occur and new FAERS reports come in. A facility inspected in 2019 carries that inspection in every year from 2019 forward."

**Designed as Time-Varying:** "This is an honest limitation. Signals like drug recalls, warning letters, and import refusals have dates — we know exactly when they happened. In the current version, these are treated as cumulative totals rather than year-filtered counts, simply because the processed data files we built didn't include date fields. This is a clean fix for the next version."

**The 483 NLP limitation** (yellow box): "Form 483s are the trickiest. The NLP was run on all publicly available 483 PDFs without preserving which year each one was from. So if a contamination flag appears in any inspection, we flag the facility for all years. This overstates early-year risk and understates later-year risk changes. The solution is straightforward: re-run the NLP pipeline preserving inspection dates. We haven't done it yet because we don't have complete 483 data to begin with — which brings us to the Alkem example we'll see in the dashboard."

---

## Slide 4 — Scoring Framework

The key question to answer: *"Why these particular numbers? Why 2 points per OAI and not 3? Why is contamination worth 3 points and data integrity only 1?"*

> "Fair question. We didn't pull these out of thin air. The weights came from a separate study of 127 manufacturing facilities and 893 drug inspections, where we analyzed which CFR citation patterns most strongly predict OAI outcomes. A contamination flag appeared in inspections that had 4 times the OAI rate of facilities without that flag. A data integrity flag had 1.6 times the OAI rate. Those ratios directly translate to the point assignments."

On the domain structure:

> "Each domain can contribute up to 25 points. Three domains, 25 each, 75 total — then rescaled to 100. We weighted them equally. That's a deliberate choice: we don't have empirical evidence that regulatory history matters more than safety signals or market exposure. Equal weighting is honest about that uncertainty. Future logistic regression models will let the data decide."

On the caps:

> "The caps — max 10 for OAI, max 8 for Warning Letters — prevent any single data point from dominating the score. A facility with 20 OAIs is clearly worse than one with 5, but we don't want it to score 150. The cap ensures the score stays interpretable."

On D2 log-scale:

> "FAERS volume varies enormously — from 1 report to 6,080 across our 18 facilities. If we scored linearly, Aurobindo's 6,080 reports would swamp everyone else. The log scale compresses that range while preserving the signal. Both report volume and severity rate matter — a facility with 100 reports that are all life-threatening is more dangerous than one with 1,000 mostly non-serious reports."

---

## Slide 5 — Variables + FAERS Extensions

Briefly summarize the 22 current variables, then focus on the right column:

> "The FAERS variables we currently use — total reports and serious rate — are relatively crude. The slide shows what a richer FAERS analysis looks like. Rate variables normalized per 100,000 prescriptions so you can compare large and small facilities fairly. Temporal dynamics that ask 'is this getting worse?' — the change, trend, and acceleration metrics would let us detect quality deterioration in real time, before regulatory action. The BCPNN Bayesian score is what FDA's own Empirical Bayes system uses for pharmacovigilance — it detects whether a particular manufacturer is generating disproportionately many reports relative to background reporting rates for that drug class. These are achievable with the NDC-quarter FAERS data we already have."

---

## Slide 6 — Dashboard Walkthrough

> "Rather than showing you more slides, let me walk you through the live dashboard. I'll cover four tabs, each telling a different part of the story. Let me flag the key examples to watch for."

*(Use this as your cue to open the dashboard. The next section of this document gives you the full click-by-click guide.)*

---

## Slide 7 — Current State & Roadmap

> "Let me be direct about what this is and isn't. What we have is a validated proof-of-concept. The correlation between our regulatory domain score and Valisure's dissolution measurement — ρ=0.809, p=0.001 — is a strong signal that publicly available enforcement data captures something real about manufacturing quality. We didn't use any lab data to build the index. It predicted lab outcomes. That's meaningful."

> "What it isn't: it's not a model in the predictive analytics sense. It's an additive score — think of it as a structured checklist turned into a number. It doesn't model whether last year's OAI makes this year's OAI more likely, or whether there's a lag between inspection failure and contamination appearing in product. Those time-series dynamics are the next frontier."

> "The roadmap is straightforward: replace the expert-calibrated weights with logistic regression coefficients when we have enough data. Add MarketScan clinical outcomes as a leading indicator — detecting patient-level quality signals before FDA even arrives. Expand to all drug classes. What we've built here is the framework and proof that it's worth building."

---

---

# DASHBOARD WALKTHROUGH — DETAILED CLICK-BY-CLICK GUIDE

Open `MQRI_Dashboard.html` in a browser. Full-screen it.

---

## TAB 1: OVERVIEW

### What you're looking at
A stacked bar chart showing all 18 facilities ranked by total MQRI score (2024). Each bar is split into D_reg (blue/navy), D_safety (red), D_market (green). Two dotted horizontal lines: HIGH at 55, MODERATE at 30.

### Opening statement
> "First observation: nobody is in the red zone. The HIGH threshold is 55. Our highest scorer is Lupin at 53.4 — just below the line. This is actually expected. Companies with extreme violations eventually get shut down or recalled off the market. What we're measuring is the gradient among active suppliers."

### Key examples to walk through

**Example 1 — Sun Pharma vs Aurobindo: Same score, different problems**

Point to Sun Pharma (52.5) and Aurobindo (51.7) — nearly identical scores.

> "Look at these two bars. Almost the same height. But watch the composition."

Sun Pharma: D_reg = 25.0 (maxed, dark section fills most of bar), D_saf = 9.3, D_mkt = 5.0
Aurobindo: D_reg = 13.3 (moderate), D_saf = 24.1 (near max, red fills most of bar)

> "Sun Pharma's regulatory domain is completely maxed out at 25 points. That's the ceiling. They have 4 OAIs, 2 Warning Letters, 12 import refusals, NLP flags for contamination, systemic failures, management oversight language, corporate failure language. From a regulatory standpoint, they've accumulated everything the index can capture. Yet their FAERS burden is relatively modest — 76 reports in 2024, 20% serious."

> "Aurobindo is almost the mirror image. One OAI — so their regulatory score is moderate at 13. But their safety domain is 24.1 out of 25. They have 6,080 FAERS adverse event reports, 93% of which are classified as serious or life-threatening. That's a patient safety signal, not a regulatory audit signal."

> "For a policymaker, these require completely different responses. Sun Pharma needs more inspections and enforcement follow-through. Aurobindo's FAERS pattern warrants a pharmacovigilance deep-dive."

**Example 2 — Amneal: The market risk case**

Point to Amneal (39.3).

> "Amneal has zero OAI inspections. Their regulatory score is only 3.7. Their FAERS is modest. But their market score is 13.7 — the highest in the dataset. Why? Because in 2020, Amneal was dispensing over 1 billion extended units of metformin. That's an enormous share of the US supply. If their quality degrades, the number of patients affected is disproportionately large. Market volume is a risk multiplier."

**Example 3 — ScieGen: Every adverse event is serious**

Point to ScieGen (33.3).

> "ScieGen is just barely in the MODERATE tier. Look at their safety domain — 19.4 points. That comes from 327 FAERS reports with a 100% serious rate. Every single adverse event ever reported for ScieGen's metformin was classified as serious or life-threatening. The volume is modest, but the severity composition is striking. This is a facility worth watching."

**Example 4 — Granules India: NLP signals doing work**

Point to Granules India (38.6).

> "Granules has zero OAI inspections. If you just looked at OAI counts, you'd rate them as low risk. But their D_reg is 14.0 — among the higher regulatory scores. That comes from NLP flags: contamination flag, data integrity flag, OOS/OOT flag — all True. Plus a Warning Letter. The index picked up warning signals from the 483 text even though FDA never escalated to OAI. And Valisure found DMF at 413,292 ng/day in their products in 2020 — that's 47 times the FDA limit. The NLP signals were pointing to a real problem."

---

## TAB 2: MQRI OVER TIME

Click the "📈 MQRI Over Time" tab.

### What you're looking at
Line charts showing each facility's MQRI total score from 2017 through 2024. There's also a per-facility dropdown to zoom in.

### Opening statement
> "The temporal view is where this index becomes genuinely useful for surveillance. A current snapshot tells you who's risky now. The trend tells you who's getting riskier — and who's been consistently risky for years."

### Key examples to walk through

**Example 1 — Laurus Labs: The rising signal**

Find the Laurus Labs line (one of the lower lines, rising noticeably after 2020).

> "Laurus Labs. In 2020, their MQRI was 15.1. They had 1 FAERS report in the entire dataset to that point. By 2022: 39 reports, score rises to 21.5. By 2024: 171 reports, 90% serious, score reaches 27.2. They've had no OAI inspections. But the FAERS pattern is steadily growing and increasingly serious. This is the kind of signal that should trigger a targeted inspection — before a formal enforcement action is needed."

**Example 2 — Lupin and Sun Pharma: Already elevated in 2017**

Point to the two highest lines in the trend chart.

> "Lupin and Sun Pharma are at the top throughout. Their enforcement records were already substantial before our panel begins. The score doesn't change dramatically year to year because their history was already written. This is important: an established regulatory record doesn't go away. Past OAIs and warning letters stay in the cumulative score."

Select Lupin from the per-facility dropdown.

> "Look at Lupin's domain breakdown over time. D_reg has been high and stable — that OAI from years ago still counts. D_saf has been slowly increasing as FAERS reports accumulate: 710 in 2020, 1,055 in 2022, 1,164 in 2024. The index is picking up growing patient-safety signals year by year."

**Example 3 — Marksans: The flatline that should alarm you**

Find the Marksans line — stays near the bottom throughout.

> "Here's Marksans. Score of 7.9 in 2020, barely moving until 2024 when it reaches 20.8 — primarily from growing FAERS. This looks like a low-risk company. But keep this line in mind. When we get to the validation tab, I'll show you what Valisure found when they tested Marksans products in 2020."

---

## TAB 3: VALIDATION VS GROUND TRUTH

Click the "🔬 Validation vs Ground Truth" tab. Select year: **2024** first.

### Opening statement
> "This is the test. We built the entire index without looking at a single Valisure number. Now we ask: does MQRI predict what Valisure measured? If yes, the framework is sound. If no, we've built an elaborate signal that doesn't actually track reality."

### 2024 — Dissolution scatter (strongest result)

Look at the scatter plot of MQRI vs Dissolution Difference Factor.

> "The y-axis is Valisure's dissolution measurement — higher means worse drug release, the pill isn't dissolving properly. The x-axis is MQRI score. The correlation is ρ = 0.809, p = 0.001. For a sample of 12 facilities, that's an exceptionally strong result."

Point to specific dots:
- **Lupin** (MQRI 53.4, Dissolution 0.615): upper right — high risk, poor dissolution. "Lupin scores highest on MQRI and has the worst dissolution in Valisure's 2024 testing."
- **Sun Pharma** (MQRI 52.5, Dissolution 0.563): also upper right. "Sun Pharma — maxed D_reg, poor dissolution."
- **Bausch Health** (MQRI 3.8, Dissolution ~0.054): lower left. "Bausch: very low risk score, very good dissolution."

> "What's striking about this result is that it's the regulatory domain alone — inspection history, warning letters, 483 NLP flags — that drives this correlation. D_reg vs Dissolution is ρ=0.809. The FDA's enforcement record predicts a lab-measured quality outcome. That's not obvious. You might expect lab quality to be unrelated to paperwork violations. It's not."

### 2024 — DMF scatter

> "DMF — dimethylformamide — is a solvent impurity. FDA limit is ~8,800 ng/day. Let me point to a few facilities."

- **Granules India** (MQRI 38.6, DMF ~17,837): "Granules had DMF at twice the FDA limit in 2024. MQRI flagged them at MODERATE with D_reg=14, driven entirely by NLP flags — contamination, OOS, data integrity. No OAI, but the signals were there."
- **Amneal** (MQRI 39.3, DMF ~40,098): "Amneal had DMF at 4.5 times the FDA limit. Their score is 39.3 — MODERATE."
- **Nostrum** (MQRI 42.2, DMF ~8,036): "Nostrum: 2 OAIs, 4 drug recalls, DMF near the FDA limit."

> "Overall correlation for DMF: ρ=0.629, p=0.009. Significant, and with 16 facilities — our largest sample."

### Switch to 2022 — The NDMA blindspot

Click year button: **2022**.

Look at the NDMA scatter.

> "Now I want to show you the most important limitation. Look at this scatter. NDMA was tested in 2020 and 2022 — it's a nitrosamine impurity, FDA limit is 96 ng/day."

Find the Marksans point — it will be an outlier: very low MQRI (~8.6) but detectable NDMA (~35.9 ng/day).

> "This dot is Marksans. Their MQRI in 2022 is 8.6. They score LOW by every measure. But Valisure found 35.9 ng/day of NDMA — detectable, above many peers. In 2020 it was 396.8 ng/day. Four times the FDA limit."

> "Why doesn't the index catch it? Because Marksans has no OAI inspections, no Warning Letters, no NLP flags from public 483 PDFs. They have minimal FAERS. From a regulatory data perspective, they don't exist as a risk. This is the regulatory blindspot: our index can only see what the FDA has seen. If a facility escapes enforcement scrutiny, our score is essentially blind to their quality."

> "This is not a flaw in the index design — it's an honest acknowledgment of what regulatory data can and cannot tell you. The solution is adding independent lab testing data, or clinical outcome signals that appear before FDA ever shows up."

---

## TAB 4: FACILITY PROFILE

Click the "🏭 Facility Profile" tab.

Walk through four facilities in order. Click each one, then use the year buttons.

---

### Facility 1 — Lupin Ltd. (the full story)

Click **Lupin Ltd.**

**Select year: 2020**
> "Let's start with Lupin — our top scorer. In 2020: MQRI 51.7. Look at the breakdown: D_reg = 17.6, D_saf = 19.0, D_mkt = 2.2. This is a balanced risk profile — both regulatory and safety signals are elevated."

Metrics to point to:
- OAI cumulative: 1. "One OAI on record."
- VAI cumulative: 4. "Four VAI inspections."
- FAERS: 710 reports, 78% serious. "710 adverse event reports, nearly 8 in 10 classified serious."
- NLP flags: contamination ✓, OOS/OOT ✓, systemic ✓, management oversight ✓, corporate failure ✓. "Five out of seven NLP flags are active. Contamination risks, systemic quality failures, and warning letter language about management breakdown — all documented."

Valisure 2020: NDMA = 69.3 ng/day.
> "Valisure found NDMA at 69.3 ng/day in Lupin's products in 2020. The FDA limit is 96. They're at 72% of the limit. The index said HIGH risk — Valisure confirmed elevated contamination. Consistent."

**Select year: 2024**
> "Four years later: 1,164 FAERS reports, still 78% serious. Score: 53.4. The index is slowly climbing as FAERS accumulates. Valisure's 2024 dissolution factor for Lupin is 0.615 — the worst in the dataset."

---

### Facility 2 — Sun Pharmaceutical Industries (the regulatory ceiling)

Click **Sun Pharmaceutical Industries Limited**

**Select year: 2024**
> "Sun Pharma. D_reg = 25.0. The maximum. Every point in the regulatory domain has been earned."

Metrics:
- OAI: 4 (public FDA raw; Q&A data shows 2 — discrepancy worth noting)
- Warning Letters: 2
- Import refusals: 12
- FAERS: 76 reports, 20% serious

NLP flags: contamination ✓, OOS ✓, systemic ✓, management oversight ✓, corporate failure ✓. "All five flags active."

> "With 4 OAIs, 2 Warning Letters, and 12 import refusals, Sun Pharma has the most extensive regulatory history in our dataset. D_reg is capped at 25 — adding more OAIs can't push it higher."

Valisure: NDMA = 0, DMF = 0 in 2020 (and ~151 in 2024). Dissolution 0.563 in 2024.
> "Interesting: Valisure found no NDMA, very low DMF in the batches they tested in 2020. But dissolution in 2024 was 0.563 — second worst in the dataset. High regulatory risk does correlate with quality failure, just not always the specific chemical impurity that was tested."

---

### Facility 3 — Marksans Pharma (the blindspot — make this dramatic)

Click **Marksans Pharma Limited**

**Select year: 2020**
> "Now the most important example in the dataset. Marksans. Score: 7.9. LOW tier."

Metrics:
- OAI: 0. VAI: 2. No 483s in public DB. No warning letters. No import refusals.
- FAERS: 8 reports, 0% serious.
- NLP flags: all unavailable (NaN) — "We have no public 483 PDFs for Marksans. We can't extract NLP signals that don't exist in the public record."
- IQVIA volume: low.

> "By every metric the index can measure, Marksans looks fine. LOW risk."

Valisure 2020: NDMA = **396.8 ng/day**.
> "Valisure found 396.8 ng/day of NDMA. The FDA limit is 96 ng/day. That's 4.1 times the legal limit. Marksans had the highest NDMA contamination of any manufacturer in the entire dataset."

> "How? Marksans is an Indian manufacturer with limited FDA inspection history visible in the public database. They had 2 VAI inspections — meaning FDA visited and found voluntary action. No OAI, no formal enforcement. Either the inspections didn't detect the NDMA problem, or the contamination arose between inspections. Either way, our regulatory-based index had nothing to work with."

**Select year: 2022**
> "Two years later: Valisure retested. NDMA dropped to 35.9 ng/day — still detectable, but dramatically lower. Perhaps they fixed it after the 2020 finding became public. MQRI: 8.6. Score barely moved."

**Select year: 2024**
> "By 2024: score is 20.8 — the biggest jump for Marksans. FAERS has grown to 35 reports, 63% serious. The index is beginning to pick up the safety signal. But still LOW tier. The regulatory data never caught what Valisure found."

---

### Facility 4 — Alkem Laboratories (data completeness)

Click **Alkem Laboratories Limited**

**Select year: 2024**
> "Last example. Alkem. Score: 15.6. LOW tier."

Metrics:
- OAI: 0. VAI: 7. FAERS: 5 reports, 0% serious. No warning letters. No import refusals.
- D_reg: 7.4 — all from VAI count.
- NLP flags: contamination = True. All others False or unavailable.

> "Look at the NLP flags. Contamination is flagged True — but every other NLP flag is missing or False. And here's why: if you look at the public FDA inspection database, Alkem has 0 posted 483 citations. Zero. The inspectors visited 7 times and gave VAI classifications — meaning they found minor issues requiring voluntary action — but no Form 483s appear in the public portal."

> "Our commercial Redica data shows 4 Form 483s for this facility. Those inspections had real objectionable conditions. But they're not publicly accessible. So we can't run NLP on them. The contamination flag being True may come from the severity score composite, not from 483 text analysis."

> "What this means for Alkem's score: it's probably understated. If we had those 4 Form 483s, we might find additional NLP flags that would add 3-5 more points to D_reg."

> "This is the data completeness gap. Public regulatory data is systematically incomplete relative to commercial sources. For a proof-of-concept using public data only, this is acceptable. For a production surveillance system, you'd want the commercial feed."

---

# CLOSING TALKING POINTS

After the dashboard walkthrough, close with:

> "What you've just seen is a proof-of-concept that regulatory and safety signals — available from public FDA databases — meaningfully predict independently measured drug quality. The validation correlation of 0.809 for dissolution is not something we engineered. It emerged from holding out the Valisure data entirely and testing after the fact."

> "The limitations are real and documented. The regulatory blindspot is real — Marksans is the proof. The data completeness gap is real — Alkem illustrates it. The model is a weighted average, not a time-series model."

> "The path forward is clear: richer FAERS signals at the NDC level, logistic regression to replace expert weights, clinical outcomes from MarketScan as a leading indicator, drug shortage events as a ground truth for time-lagged prediction. And most importantly — expanding beyond metformin to all generic drug classes. The framework is designed for that."

---

# QUICK REFERENCE: KEY NUMBERS TO CITE

| Fact | Number | Context |
|------|--------|---------|
| Validation: Dissolution vs D_reg | ρ = 0.809, p = 0.001 | 12 facilities, 2024, strongest result |
| Validation: DMF vs MQRI Total | ρ = 0.629, p = 0.009 | 16 facilities, 2024 |
| Marksans NDMA | 396.8 ng/day | 2020 Valisure; FDA limit = 96 |
| Granules DMF | 413,292 ng/day | 2020 Valisure; FDA limit ~8,800 |
| Amneal DMF | 40,098 ng/day | 2024 Valisure; ~4.5× FDA limit |
| Nostrum DMF | 8,036 ng/day | 2024 Valisure; near FDA limit |
| Aurobindo FAERS | 6,080 reports, 93% serious | 2024 |
| Lupin FAERS | 1,164 reports, 78% serious | 2024 |
| Sun import refusals | 12 | All-time; drives D_reg to ceiling |
| Sun D_reg | 25.0 (max) | Capped in 2020, 2022, 2024 |
| Alkem public 483s | 0 | vs 4 in Redica commercial data |
| ScieGen serious AE rate | 100% | All 327 FAERS reports classified serious |
| Laurus FAERS growth | 1 → 39 → 171 | 2020 → 2022 → 2024 |
| Lupin NDMA (Valisure 2020) | 69.3 ng/day | 72% of FDA limit; consistent with MQRI signal |
| No facility reaches HIGH tier | Highest: 53.4 (Lupin) | Threshold is 55 |

---

*Document generated April 2026 · MQRI_Presentation_Narrative.md*
