# INFORMS 2026 — Practice Narrative
## "From Regulatory Text to Patient Harm: LLM Extraction of FDA 483 Signals and Adverse Event Prediction"

**Total target time: ~18-20 minutes**
**Presenter: Amirreza Sahebi-Fakhrabad**

> **Pause key:** `[PAUSE]` = stop, breathe, look at audience before continuing. Longer pauses at major transitions.

---

## SLIDE 1: Title

"Good afternoon. My name is Amirreza Sahebi-Fakhrabad, I'm a PhD student at NC State's Poole College of Management, working in supply chain and healthcare operations. **[PAUSE]**

This work is a collaboration between NC State and Ohio State. My co-authors are Amir Hossein Sadeghi and Robert Handfield at NC State, and Shailesh Divey and John Gray at Ohio State. **[PAUSE]**

This work is part of a larger research program with two connected goals: understanding what drives drug manufacturing quality failures, and predicting drug shortages before they occur. Today I am focusing on the quality side -- specifically, the question driving this slide: can the text that FDA inspectors write during facility visits predict which manufacturers will produce the most patient harm? **[PAUSE]**

That is what I will show you today."

**[PAUSE -- let the framing land before clicking to slide 2]**

---

## SLIDE 2: Generic Drug Quality -- A Regulated Industry With Visibility Gaps

"Generic drugs are 90 percent of U.S. prescriptions -- that scale makes quality oversight critical. **[PAUSE]**

FDA inspects each facility every one to three years on average. Between visits, real-time insight into what is happening on the manufacturing floor is limited. **[PAUSE]**

When problems go undetected long enough, they tend to trigger recalls, warning letters, and enforcement actions that force production reductions. Manufacturing quality is the leading cause of drug shortages. **[PAUSE]**

The diagram on the right captures the basic dynamic: quality drifts during the inspection gap, and what comes out the other end is patient harm and shortage risk. **[PAUSE]**

The central challenge is catching those signals before they get that far."

---

## SLIDE 3: FDA Form 483 -- Structure and the Classification Gap

"Let me walk through how the FDA inspection process works, because it explains exactly why we built this dataset. **[PAUSE]**

When an FDA inspector finishes visiting a facility, on the last day they hand the company a Form 483 -- a written list of every GMP violation observed. GMP stands for Good Manufacturing Practice -- the federal standards covering how drugs must be manufactured, tested, and stored. The deficiencies in a 483 are primarily GMP failures: contamination controls, laboratory records, data integrity, quality system gaps. That is the document on the right. Standard header at the top, then numbered observations in the inspector's own words, specific to what was found at that facility on that day. **[PAUSE]**

After returning to the office, the inspector writes a much larger document called the Establishment Inspection Report, or EIR. The EIR is the comprehensive record -- full narrative behind each observation, supporting exhibits, batch records, management discussions, and the company's written responses to the 483. FDA then uses the EIR and those responses to assign the final classification -- NAI, VAI, or OAI -- sent to the facility later. **[PAUSE]**

Here is the data access situation. We have the 483 text, obtained via FOIA requests through Redica. The EIR is richer, but researchers do not yet have systematic access to it. The final classification is the only fully public signal. **[PAUSE]**

This also means we do not fully know how FDA arrives at the final outcome label -- the EIR and internal review process are not visible to us. The relationship between what the 483 says and what classification a facility ultimately receives is itself an open research question, especially given documented cases where facilities were reclassified from OAI down to VAI for reasons outside the inspection findings. Understanding that correlation is something we flag as future work. **[PAUSE]**

What we can say now: relying solely on those three outcome labels while ignoring the rich inspector-written text is discarding most of the information the inspection actually produced. The difference between what the label says and what the text says -- that is what this paper is about."

**[PAUSE -- major transition before study design]**

---

## SLIDE 5: Study Design -- 14 Drugs, 98 Facilities, 2018-2026

"Our sample covers 14 generic APIs spanning cardiovascular drugs, anti-infectives, and critical care medications -- Atorvastatin, Lisinopril, Metoprolol, and Metformin on the cardiovascular side; Ampicillin, Ampicillin-Sulbactam, Metronidazole, and Vancomycin for anti-infectives; and Bupropion, Calcium Gluconate, Magnesium Sulfate, Pantoprazole, Potassium Chloride, and Tacrolimus rounding out the rest. We used DailyMed product labels -- which carry manufacturer name, DUNS number, and facility location -- to trace NDC codes to the producing facility, which gave us 129 unique facilities. **[PAUSE]**

Of those 129, 98 had Form 483 text available through Redica along with inspection outcome classifications. Those 98 facilities produced 246 inspection events between 2018 and 2026. **[PAUSE]**

Our outcome variable is serious adverse events from FAERS. Each FAERS report carries an ANDA number -- we use the Orange Book crosswalk to link that ANDA back to the specific facility that manufactured the drug. That means we count AEs from that specific facility, not from everyone who makes that drug. We restrict to serious outcomes: death, hospitalization, life-threatening events, disability. 78 of the 98 facilities have ANDA-matched AE data; the remaining 20 have no FAERS reports citing their specific ANDA, so we exclude them from the outcome analysis. **[PAUSE]**

The panel has one row per inspection event, with AE counts at plus and minus four quarters. 246 rows total for text features; 176 rows with ANDA-matched outcome for the predictive model."

**[PAUSE -- transition]** *"Now let me walk you through the pipeline we built to turn 483 text into structured features."*

---

## SLIDE 7: Pipeline Diagram

"The pipeline has five steps. We start with raw 483 observation text. **[PAUSE]** An LLM extracts structured signals from each observation. **[PAUSE]** Those signals are aggregated from the observation level up to the inspection level. **[PAUSE]** We join the inspection features to our FAERS panel. And then we run correlation analysis and the predictive model. **[PAUSE]**

Let me walk through each step in a bit more detail."

---

## SLIDE 8: Step 00 -- What a 483 Observation Looks Like

"Here are two actual observations from a Sun Pharma inspection in 2022. This was an OAI inspection -- six observations. **[PAUSE -- let them read the boxes]**

Observation 1 is a data integrity finding. The header cites a failure to thoroughly investigate discrepancies. The body describes instances of backdating by QA and QC personnel. **[PAUSE]**

Observation 2 is a lab controls finding. The header cites failures to document sampling at the time of performance. The body includes evidence that an employee's building access records did not match the buildings where samples were claimed to have been collected. **[PAUSE]**

Two observations from the same inspection, covering completely different risk domains. The domain, severity, and patient risk implications are embedded in the prose -- there is no structured schema that would let you parse them automatically without reading the text. **[PAUSE]**

One question that often comes up: are these observations actually variable enough to extract useful signals, or do they all look similar because they follow a standard CFR citation format? **[PAUSE]** We computed pairwise Jaccard similarity across all 1,067 observations in our dataset. The mean is 0.10, and 99 percent of pairs fall below 0.20. The shared CFR header structure accounts for almost none of the vocabulary. Each observation is genuinely facility-specific."

---

## SLIDE 9: Step 01 -- What the LLM Extracts

"We extract two groups of signals from each observation. **[PAUSE]**

Group 1 is binary flags. Does this observation describe data integrity violations -- backdating, falsification, reconstructed records? Is there contamination? Is there a direct patient harm pathway? Did the investigator find an inadequate OOS investigation? Was this a repeat finding from a prior inspection? **[PAUSE]**

Group 2 is multi-category signals. Which of the eight CFR 211 domains does this observation fall under -- Lab Controls, Quality System, Production, and so on. **[PAUSE]** What is the severity -- Critical, Major, Moderate, or Minor. What is the root cause -- a Capital gap, meaning resource constraints, or a Cultural gap, meaning management failure. What is the scope -- a single batch, multiple products, or facility-wide. **[PAUSE]**

These dimensions were chosen to capture the aspects of an inspection finding most directly linked to patient risk, not just procedural compliance."

---

## SLIDE 10: Step 01 -- Turning Text into Structured Features

"The extraction uses Claude Haiku, one pass per observation, with a fixed JSON output schema. **[PAUSE]**

The example on the left shows what this looks like in practice. The input is: 'Batch production records were not completed at the time of manufacture. Entries were reconstructed from memory 2 days after processing.' **[PAUSE]** The model assigns severity Major and sets the data integrity flag to True, with a rationale: confirmed defect; records reconstructed. **[PAUSE]**

The JSON schema is fixed, so the model cannot invent new fields or add free-text commentary. This keeps the output clean for aggregation. **[PAUSE]**

On the trustworthiness question: Khairallah et al. (2024) tested 18 LLMs including Claude Haiku on clinical entity extraction from surgical notes and found 90.2 percent accuracy. We also had Redica GMP specialists review our prompt definitions in June. **[PAUSE]** Manual human labeling is in progress and findings should be treated as preliminary, but we have reasonable confidence the extraction is capturing what we intend."

---

## SLIDE 11: Step 02 -- From Observation Labels to Inspection Feature Vectors

"Once we have labels for each observation, we aggregate up to the inspection level. This gives us three types of features. **[PAUSE]**

Share features -- what fraction of observations in this inspection fell in a given domain or had a given flag. This tells you what kind of inspection it was. **[PAUSE]**

Count features -- the raw number of observations in a domain. This tells you how much was cited. **[PAUSE]**

Co-occurrence flags -- whether two signals appeared in the same inspection together. Lab Controls alone means a QC process failed. Data integrity alone means a record was altered. **[PAUSE]** Both together in the same inspection -- QC failures and documentation manipulation -- is a stronger combined signal than either alone. **[PAUSE]**

The output is 246 inspections by 17 features, which feeds directly into the analysis."

**[PAUSE -- major transition, this is the pivot to results]** *"Let me now turn to what those features actually predict."*

---

## SLIDE 12: Transition -- Analysis and Results

"Two questions. **[PAUSE]** Are 483 text signals associated with future patient harm? **[PAUSE]** And among VAI facilities -- where FDA's label provides no differentiation -- does text separate high-harm from low-harm producers? **[PAUSE]**"

---

## SLIDE 13: Do 483 Text Signals Carry Information?

"Here are the prediction results. The task: given 17 LLM features from a single inspection, predict whether that facility will produce above-median adverse events over the following four quarters. FEI-grouped 5-fold cross-validation, logistic regression, baseline AUC 0.5 by construction. 176 inspection events across 78 facilities with ANDA-matched AE data. **[PAUSE -- let them read the table]**

Rows A through C are the full sample -- 176 inspections across 78 facilities with ANDA-matched adverse event data. Text alone scores 0.585, and the inspection outcome flag alone scores 0.545. **[PAUSE]** Text modestly outperforms the structured label, and combining them adds nothing -- text already captures what the flag captures. **[PAUSE]**

Row D is the key result. **[PAUSE]** VAI-only facilities. Text alone. AUC of 0.656, with a p-value of 0.046 from a one-tailed t-test across the five folds against the null of 0.5. **[PAUSE]**

Let me be direct about what 0.66 means. It is not a strong classifier. You would not use this model alone to make enforcement decisions. **[PAUSE]** But it is statistically distinguishable from random in the subgroup where no other signal can differentiate facilities. That is the claim -- not that the model is powerful, but that the text carries detectable information where the inspection outcome label carries none. **[PAUSE]**

Row E shows that in OAI-ever facilities, text adds nothing and actually scores below random. That makes sense: enforcement has already acted on the worst problems. The signal lives in the silent cases."

---

## SLIDE 14: The Silent Problem -- Same VAI Label, Very Different Patient Outcomes

"Let me now show you what that signal looks like in practice, without any model involved -- just AE trajectories. **[PAUSE]**

We split VAI-only facilities into two groups based on Lab Controls share: the top quartile, which we call Hi-sig VAI, and the rest. **[PAUSE -- let them read the top table]**

Hi-sig VAI facilities start at 40 AEs per quarter at Q0. One year later, they are at 49. Lo-sig VAI facilities start at 34 and stay at 30. **[PAUSE]** Hi-sig VAI produces 63 percent more AEs at Q+4 than Lo-sig VAI. **[PAUSE]**

The OAI-ever group, for comparison, actually shows a slight decline. Enforcement had an effect. **[PAUSE]**

The bottom table shows named examples: Lupin for Atorvastatin, Dr. Reddy's for Metoprolol, Sun for Atorvastatin. All three had persistent AE increases with no OAI classification. **[PAUSE]** The 483 text from their inspections flagged the Lab Controls problems; the outcome label said VAI and that was the end of it. **[PAUSE]**

Why did enforcement not follow? One explanation, documented by investigative reporting from ProPublica and Bloomberg, is that FDA has in some cases reclassified OAI inspections to VAI under drug shortage pressure, and downgraded facilities show worse subsequent patient outcomes. **[PAUSE]** Another explanation is that these facilities stayed just below the OAI threshold or that inspectors did not surface the full severity. Either way, the label was not telling the whole story, and the text was."

---

## SLIDE 15: Summary

"Let me summarize. **[PAUSE]**

First: yes, 483 text is associated with future harm, but the signal concentrates in facilities FDA never penalized. In the full sample, text reaches AUC 0.585 and modestly outperforms the inspection outcome flag. In VAI-only facilities, text alone reaches AUC 0.656. **[PAUSE]**

Second: Lab Controls is the leading dimension. The count and severity of Lab Controls observations reach rho up to 0.29 at Q+2. No other dimension is significant at both pre- and post-inspection snapshots. **[PAUSE]**

Third: same VAI label, very different outcomes, and those outcomes are identifiable from text. Hi-sig VAI facilities produce 63 percent more AEs one year later than Lo-sig VAI. These are the silent cases -- facilities that stayed below the enforcement threshold even as patient harm accumulated."

**[PAUSE -- slow down here, let the last point settle]**

---

## SLIDE 16: Limitations

"There are important limitations to be transparent about. **[PAUSE]**

Sample size is the most significant one. 98 FEIs and 33 OAI inspections. The VAI-only results are hypothesis-generating, not confirmatory. We need more data and ideally an out-of-sample test. **[PAUSE]**

COVID disrupted the inspection schedule significantly. FDA suspended most inspections in 2020 and 2021, leaving multi-year gaps. Post-2022 baselines are confounded by backlog inspections. **[PAUSE]**

FAERS attribution has a known limitation: when one manufacturer operates multiple facilities under the same ANDA, we cannot determine which site produced the drug in any given report. **[PAUSE]**

LLM extraction quality is still being validated. Human labeling is in progress. Findings should be treated as preliminary. **[PAUSE]**

We also have not integrated commercial volume. Facility size is a potential confounder. And finally, all associations are correlational. Testing the shortage-pressure hypothesis properly requires market concentration data we do not currently have."

---

## Q&A SLIDE

"Thank you. I'm happy to take questions."

---

## Q&A PREPARATION: GENERAL

### Q: "Is 0.585 actually better than random? Isn't that a very low AUC?"

"You're right that 0.585 is not a strong classifier -- I want to be clear about that. If you needed to use this model operationally to make enforcement decisions, that AUC is not sufficient.

But the framing I'd offer is this. The VAI-only subgroup is the hardest possible test. Every facility in it has the same FDA label. The outcome label contains zero information to differentiate them. Text-only logistic regression reaches 0.656 in that subgroup, with a p-value of 0.046 across five cross-validation folds. In a setting where the null is truly 0.5 by construction and the only available signal is inspector-written text, that is a meaningful result.

The claim is not 'this model is a good classifier.' The claim is 'this text carries information that the enforcement system is not currently using, even in the cases where text is the only differentiating signal available.'"

### Q: "Can't the SSM already use 483 text? Maybe FDA already incorporates this."

"The SSM methodology is not public -- we cannot claim with certainty what inputs it uses. What we can say is that the SSM inputs described in FDA's public documents are inspection outcome classifications, recall history, patient exposure, and time since last inspection. The 483 observation text is FOIA-only and is not part of any publicly described input to the SSM. Whether FDA uses it internally, we cannot confirm."

### Q: "Why use FAERS? It has serious reporting biases."

"FAERS has well-known limitations: underreporting, spontaneous reporting variability, and the inability to definitively establish causality. We restrict to serious outcomes only, which have mandatory reporting requirements and more consistent documentation. We also use raw AE counts rather than rates, precisely because our volume denominator only covers about 69 percent of FEI-years, and dividing by a partially missing denominator would introduce systematic bias.

The longer-term plan is to incorporate MarketScan, which has more consistent capture of drug dispensing and patient outcomes. But FAERS is the only data source that currently provides the ANDA-level attribution we need to link an adverse event to the producing facility."

### Q: "Why only 14 drugs?"

"The 14 drugs were selected because they are independently tested by a third-party lab, which gives us a ground-truth quality signal that anchors a parallel validation analysis. For this specific paper, the 14-drug scope is a practical constraint of what 483 text is available for through our data access. Expanding the drug scope is part of the next phase."

### Q: "Could the AE rise before inspections just reflect FDA targeting -- they go to bad facilities?"

"That is exactly what the inspection gap analysis in our backup slides addresses. When we condition on facilities where the previous inspection was one to two years ago, we have a clean baseline -- AEs were suppressed after the prior visit and we can measure the rise over the gap. In that group, AEs rise 37 percent before FDA arrives. That is real quality drift during the inter-inspection window, not a targeting artifact. And the 483 text from the inspection at the end of that window is what we use as our feature set."

---

## Q&A PREPARATION: LLM-SPECIFIC QUESTIONS

*These are the questions most likely to come from an audience that is not familiar with LLMs. Prepare honest, simple answers -- do not overclaim.*

---

### Q: "Why did you use an LLM at all? Couldn't you do this with keyword matching or regular expressions?"

"That was actually our starting point. We tried keyword-based approaches first. The problem is that the same concept appears in very different forms across observations. 'Backdating' is sometimes written as 'reconstructed records,' 'entries completed after the fact,' or 'documentation prepared retrospectively.' A keyword list either misses many of those or generates too many false positives from unrelated mentions of the same words.

The LLM reads the full context of the observation and makes a judgment about meaning, which is what a human reviewer would do. For highly variable, narrative-style regulatory text, that flexibility is the main advantage."

### Q: "Why Claude Haiku specifically? Did you try other models?"

"We chose Claude Haiku because it is a smaller, faster model well-suited for structured extraction tasks where the instructions are explicit and the output schema is fixed. We are not asking it to reason through complex ambiguous problems -- we are asking it to read a paragraph and fill in a JSON form. For that task, a smaller model with a well-defined prompt performs comparably to larger models at a fraction of the cost.

We did not run a full benchmark across multiple models for this version of the work. That is something we plan to do in the next phase, and the Khairallah et al. 2024 paper I cited provides some external benchmarking evidence that supports using LLMs for this class of task."

### Q: "How do you know the LLM is not hallucinating or making up classifications?"

"Two things constrain hallucination here. First, the output schema is fixed. The model can only return values we define -- for severity, it can only return Critical, Major, Moderate, or Minor. It cannot invent a new category. Second, every extracted value is accompanied by a short rationale that we can audit. We can check whether the rationale actually refers to something in the observation text.

That said, I want to be direct: we have not yet completed a full human validation of the extracted labels. That work is in progress. What we have done is a review by Redica GMP specialists who confirmed the prompt definitions align with industry practice. We are treating these findings as preliminary until the human labeling is complete."

### Q: "How sensitive are the results to how you wrote the prompt? Could different wording give very different results?"

"That is a legitimate concern and one of the standard critiques of LLM-based extraction. We designed the prompts with explicit definitions for each category -- for example, severity Critical requires confirmed product impact with patient exposure, not just a systemic risk finding. The goal was to reduce ambiguity in the definition so the model applies it consistently.

We have not done a formal prompt sensitivity analysis -- that is a gap. What I can say is that the results are consistent with what domain experts expect qualitatively: Lab Controls and severity are the dimensions most directly linked to product failure risk, and those are the ones that show up as significant in the correlations. If the prompt were badly specified, we would expect noisier or inconsistent results."

### Q: "Is the extraction reproducible? If you ran the same observation through the model twice, would you get the same answer?"

"For extraction tasks with a fixed schema and low-ambiguity definitions, LLMs are quite consistent. We ran each observation once with temperature set to zero, which makes the output deterministic given the same input. So yes, re-running the extraction on the same data would produce the same results.

The deeper question is whether a different annotator -- human or model -- would agree with our labels. That is what the ongoing human validation study is designed to measure."

### Q: "Couldn't a simpler NLP method like BERT or a text classifier do the same job?"

"It could potentially, but it would require labeled training data. A fine-tuned BERT-style classifier needs hundreds to thousands of human-labeled examples per dimension before it generalizes reliably. We have 1,067 observations and 10 dimensions to label. Getting sufficient labeled data for supervised training would itself be a major research effort.

The LLM approach lets us get signal from the full corpus without pre-labeled training data, using prompt definitions that domain experts can directly review and adjust. The tradeoff is that validation is harder -- which is exactly why we are running the human labeling study in parallel."

### Q: "What if the LLM gets a dimension wrong for a specific observation? Does one mistake affect the whole result?"

"Individual observation errors get partially absorbed by the aggregation step. We aggregate from 1,067 observations up to 246 inspection events. If the model mislabels one or two observations out of the five or eight in a given inspection, the share features -- which are fractions -- shift by a small amount. The co-occurrence and count features are more sensitive to individual errors, which is one reason the share features tend to be more robust in the correlations.

That said, systematic bias in the extraction -- for example, if the model consistently misidentifies a particular domain -- would carry through to the results. That is another reason we treat the current findings as preliminary."

### Q: "How much does it cost to run this at scale? Would it be practical for FDA to use?"

"For this dataset, 1,067 observations at roughly 300-500 tokens each, the total extraction cost was under ten dollars using a small commercial LLM. That is a one-time cost to process eight years of 483 data for 98 facilities.

For FDA's full dataset, which includes tens of thousands of inspections, the cost scales linearly but remains quite manageable -- we are talking about hundreds of dollars, not hundreds of thousands. The bottleneck is not cost; it is data access and validation. The 483 PDFs have to be FOIA'd and processed, and any deployment in a regulatory context would require extensive validation against human expert labels."

### Q: "Did you do any fine-tuning of the LLM, or is this purely off-the-shelf?"

"Purely off-the-shelf, no fine-tuning. The model is used exactly as provided, with a structured prompt and a fixed output schema. Fine-tuning would require labeled training data that we do not yet have in sufficient quantity. The plan is to use the human labeling study to first validate the current extraction, and then potentially use those labels as training data for a fine-tuned or smaller specialized model in a future version."

---

*Practice tip: For LLM questions, the honest answer is almost always the right one. You are not claiming the LLM is perfect -- you are claiming it gives you a usable first pass that is better than nothing and that you are validating it. Do not overdefend. The two most defensible points are: (1) the fixed schema constrains hallucination, and (2) the aggregation step absorbs individual errors. If someone presses hard on validation, acknowledge it directly: "You are right that this is a limitation, and that is exactly why the human labeling study is the next step."*

---

*Overall practice tip: The two most likely challenge points are slide 13 (AUC defensibility) and slide 14 (why no enforcement). Have the two-step answer ready: 13 establishes statistical signal in the hardest subgroup, 14 shows what that signal looks like in practice without requiring the audience to interpret AUC. Lead with the 63% trajectory number if the AUC question gets hostile.*
