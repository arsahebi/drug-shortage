# Literature Review: NLP/LLM on FDA Text, Inspection-to-Quality Linkages, and FAERS as Manufacturing Quality Outcome

**Prepared:** 2026-07-06  
**Purpose:** Supports a paper using LLM-extracted features from FDA Form 483 inspection observations to predict facility-level FAERS adverse event counts as a manufacturing quality outcome.  
**Coverage:** 27 papers across three clusters.

---

## Overview

This review synthesizes literature across three interlocking clusters that together frame our contribution:

| Cluster | Count | Core Question |
|---------|-------|---------------|
| 1 — NLP/LLM on FDA Regulatory Text | 11 | Can machine learning extract structured signals from FDA inspection and label documents? |
| 2 — Inspection Outcomes → Drug Quality Events | 9 | Do FDA inspection findings predict downstream drug quality failures (recalls, shortages)? |
| 3 — FAERS as Manufacturing Quality Outcome | 7 | Can adverse event counts serve as a facility-level quality proxy? |

No prior paper applies NLP/LLM extraction to FDA Form 483 observation text and then directly uses the extracted features to predict facility-level FAERS counts. Each cluster has rich prior work but the three strands have never been combined.

---

## Cluster 1 — NLP/LLM on FDA Regulatory Text

### Paper Table

| Title | Authors | Year | Journal | Relevance | Key Finding |
|-------|---------|------|---------|-----------|-------------|
| Scaling medical device regulatory science using large language models | Li, He, Subbaswamy, Vossler, Gossmann, Singh, Feng | 2026 | NPJ Digital Medicine | 3 | LLMs extract structured data from FDA MDRs and pre-market submissions with 93–97% agreement vs. human annotators; enables analyses previously requiring months of manual work |
| A Quantitative Study of US FDA Inspection Data for Drug Manufacturing Sites | Pazhayattil, Ingram, Sayeed | 2020 | Therapeutic Innovation & Regulatory Science | 3 | First systematic quantitative analysis of Form 483 observation data (2014–2018); trend: more 483 forms issued but fewer observations per inspection; identifies inter-correlated regulatory domains |
| BERT-Based NLP of Drug Labeling Documents: Classifying DILI Risk | Wu, Liu, Liu, Chen, Tong | 2021 | Frontiers in Artificial Intelligence | 2 | BERT applied to FDA drug labels for drug-induced liver injury classification; MCC 0.84–0.87 vs. 0.60 for keywords; portable across FDA and EMA |
| ADE Eval: Evaluation of NLP Systems for Adverse Event Extraction from Drug Labels | Bayer, Clark, Dang, Aberdeen, Brajovic, Swank, Hirschman, Ball | 2021 | Drug Safety | 2 | FDA-cosponsored shared task; 23 NLP systems evaluated; best F1=0.89 for mention-finding; concludes NLP not yet fully autonomous but suitable with human oversight |
| OnSIDES: Extracting Adverse Drug Events from Drug Labels Using NLP | Tatonetti et al. | 2025 | Med (Cell Press) | 2 | PubMedBERT fine-tuned on 200 curated labels; F1=0.90, AUROC=0.92; 7.1 M drug-ADE pairs from 47,211 labels across FDA, EMA, EMC, KEGG |
| FDA Warning Letters: A Retrospective Analysis 2010–2020 | Rathore, Li, Chhabra, Lohiya | 2022 | Journal of Pharmaceutical Innovation | 2 | 6,830 warning letters analyzed; CGMP violations: process validation (26%), data integrity (21%), QC (15%); >65% of pharma WLs to Asian manufacturers |
| Trends in FDA Data Integrity Enforcement: Analysis of 1766 Warning Letters (2016–2023) | Park, Kwon | 2025 | Therapeutic Innovation & Regulatory Science | 2 | ALCOA+ violation classification framework; post-pandemic increase in violations per company; year-over-year increases in availability and completeness violations after 2020 |
| An Analysis of FDA Warning Letter Citations from 2019–2023 | Kwiecinski | 2024 | Journal of Pharmaceutical Innovation | 2 | Citation-level trend analysis; 43% increase in warning-letter issuance rate 2019–2023; laboratory controls and quality systems consistently the top cited CFR sections |
| Empirical Study on LLM-based Classification of Requirements-related Provisions in Food-safety Regulations | Hassani, Sabetzadeh, Amyot | 2025 | arXiv:2501.14683 | 2 | GPT-4o achieves 89% precision / 87% recall classifying regulatory text provisions; fine-tuned BERT comparable; models generalize from Canadian to US regulatory text |
| RecallRisk-BERT: Multi-Task Framework for Post-Report Medical Device Recall Triage | Atalay, Yigit-Sert | 2026 | arXiv:2606.27174 | 2 | Multi-task BERT + LightGBM on 54,165 FDA device recall narratives; accuracy 0.963, macro-F1 0.856; simultaneously predicts recall severity and root-cause category |
| DART: Structured Dataset of Regulatory Drug Documents for Clinical NLP | Barone, Laudante, Riccio, Romano, Postiglione, Moscato | 2025 | arXiv:2510.18475 / ITADATA 2025 | 1 | Italian AIFA regulatory documents; LLMs validate pharmacological interaction inference; illustrates regulatory NLP expanding beyond English-language FDA corpora |

### Cluster 1 Synthesis

The dominant application of NLP to FDA regulatory text has been drug label mining — adverse drug reaction extraction from Structured Product Labels (OnSIDES, ADE Eval), drug-induced liver injury classification (Wu et al. 2021), and pharmacological interaction inference (DART). More recently, the Li et al. 2026 paper on medical device regulatory science demonstrated that GPT-4-class LLMs can extract structured data from unstructured FDA adverse-event MDRs and pre-market summaries at near-human accuracy and at scale. The inspection-text strand is less developed: Pazhayattil et al. 2020 performed the first systematic quantitative study of Form 483 observation data but used simple frequency counts from a pre-structured database rather than natural language processing of the raw observation narratives. Warning-letter text has received some attention (Rathore et al. 2022; Park & Kwon 2025; Kwiecinski 2024) but only for descriptive trend analysis, not feature extraction for prediction. The regulatory NLP literature (Hassani et al. 2025; RecallRisk-BERT) shows that transformer-based models generalize well to domain-specific regulatory language.

---

## Cluster 2 — Inspection Outcomes → Drug Quality Events

### Paper Table

| Title | Authors | Year | Journal | Relevance | Key Finding |
|-------|---------|------|---------|-----------|-------------|
| Obligatory Responses to FDA Inspection Outcomes and Future Drug Shortages | Wang (Iris), Ball, Anand, Park | 2025 | Manufacturing & Service Operations Management | 3 | IV analysis on US drug manufacturing facilities; OAI outcome → 96.4% lower shortage likelihood; VAI → 77.2% lower shortage likelihood in next 12 months; contradicts GAO 2016 report |
| Drug Shortages, Pricing, and Regulatory Activity | Stomberg | 2017/2018 | NBER Working Paper W22912 / Univ. Chicago Press | 3 | Pooled dynamic regression on FDA inspection and citation data; statistically significant relationship between regulatory activity intensity and drug shortage rates; foundational empirical paper |
| Evaluating quality reward and other interventions to mitigate US drug shortages | Naumov et al. | 2025 | Journal of Operations Management | 2 | System dynamics model of generic drug market; quality reward intervention sustainably reduces shortages vs. short-term capacity nudges; monopoly emergence risk requires quality disclosure mechanism |
| Alleviating Drug Shortages: The Role of Mandated Reporting Induced Operational Transparency | Lee, Lee, Shin, Krishnan | 2021 | Management Science | 2 | US (2012) and Canada (2017) mandatory manufacturing-disruption reporting mandates reduce time-to-recovery and annual days of shortage; effect strongest under duopoly competition |
| Development and validation of a predictive model to predict and manage drug shortages | Liu, Colmenares, Tak, Vest, Clark, Oertel, Pappas | 2021 | American Journal of Health-System Pharmacy | 2 | Logistic regression on 1,517 hospital pharmacy observations; C-statistic=0.93; IV-only formulation, generic-only status, fewer manufacturers are strongest shortage predictors |
| Predicting drug shortages using pharmacy data and machine learning | Pall, Gauthier, Auer, Mowaswes | 2023 | Health Care Management Science | 2 | XGBoost on 22 Canadian pharmacy sales records; 69% accuracy one month ahead for 784 drugs; historical shortage patterns dominate feature importance; no manufacturing quality features used |
| When Should the FDA Inspect Pharmaceutical Manufacturing Facilities to Better Mitigate Drug Shortages? | Kosmas, Ergun | 2023 | arXiv:2310.15146 | 2 | POMDP optimization model for inspection timing; quadratic relationship between inspection interval and expected value; recommends prioritizing high-risk facilities producing critical medications |
| A Machine Learning Algorithm to Predict Medical Device Recall by the FDA | Slivinskis, Maluli, Broder | 2024 | Western Journal of Emergency Medicine | 1 | Random forest on Google Trends + PubMed queries; 75–90% sensitivity, 100% specificity for device recall prediction 3–12 months ahead; methodology transferable to drug recall |
| Impact of Drug Shortages on Patient Safety and Pharmacy Operation Costs | Shaban, Maurer, Willborn | 2018 | Federal Practitioner | 1 | VA system data; shortage-related acquisition costs $150K–$750K per facility per year; clinical substitutions create patient safety risks; documents downstream harm of supply failures |

### Cluster 2 Synthesis

The empirical literature firmly establishes that FDA inspection outcomes are correlated with drug shortage risk. Stomberg (2017) provided the first regression-based evidence of a significant relationship between regulatory activity and shortage rates. Wang et al. (2025) refined this with instrumental variable analysis and showed that, counterintuitively, more severe inspection outcomes (OAI) predict *fewer* future shortages — consistent with the interpretation that inspections trigger corrective actions that ultimately stabilize supply. Machine learning drug shortage prediction (Liu et al. 2021; Pall et al. 2023) consistently achieves strong predictive performance (C-stat ≥ 0.69) but relies on demand-side pharmacy data and drug characteristics, not on inspection text signals or manufacturer quality features. No existing shortage prediction model incorporates NLP-derived features from 483 observation text. The operations management literature (Naumov et al. 2025; Lee et al. 2021) addresses policy levers but does not model facility-specific quality signals.

---

## Cluster 3 — FAERS as Manufacturing Quality Outcome Variable

### Paper Table

| Title | Authors | Year | Journal | Relevance | Key Finding |
|-------|---------|------|---------|-----------|-------------|
| Monitoring the manufacturing and quality of medicines: a fundamental task of pharmacovigilance | Sardella, Belcher, Lungu et al. | 2021 | Therapeutic Advances in Drug Safety | 3 | Conceptual and case-study argument that pharmacovigilance/AE monitoring is the primary mechanism for detecting manufacturing-origin safety failures post-approval; illustrates nitrosamine contamination and heparin scandal detection via AE signal clustering |
| A Call to Action to Track Generic Drug Quality Using Real-World Data and the FDA's Sentinel Initiative | Brown | 2020 | Journal of Managed Care & Specialty Pharmacy | 3 | Argues FAERS and Sentinel data should be systematically used to track manufacturer-level quality differences; FDA already incorporates FAERS "hazard signals" into its facility site-selection model for inspections; demonstrates two FDA-funded pilot studies showing manufacturer-level signal differences |
| Methodological considerations for comparison of brand vs. generic vs. authorized generic FAERS reports | Rahman, Alatawi, Cheng, Qian, Peissig, Berg, Page, Hansen | 2017 | Clinical Drug Investigation | 3 | Establishes methodology for using FAERS to detect quality differences across drug manufacturers via authorized-generic-as-control design; identifies and quantifies perception bias against generics; supports using FAERS for manufacturer-level quality signal detection |
| Brand vs. Generic Adverse Event Reporting Patterns: Authorized Generic-Controlled Evaluation | Alatawi, Rahman, Cheng, Qian, Peissig, Berg, Page, Hansen | 2017/2018 | Journal of Clinical Pharmacy and Therapeutics | 2 | Four cardiovascular drugs; FAERS 2004–2015; reporting pattern differences between brand and generic largely attributable to perception bias; authorized generic controls limit confounding; demonstrates FAERS-based comparative manufacturer surveillance methodology |
| Comparison of Brand vs. Generic Antiepileptic Drug Adverse Event Reporting Rates in FAERS | Rahman, Alatawi, Cheng et al. | 2017 | Epilepsy Research | 2 | 46,177 FAERS reports for 3 AEDs; brand and generic reporting similar after controlling for perception bias via authorized-generic comparison; disproportionate suicide/suicidal ideation signals for generic lamotrigine warrant further investigation |
| FDA Adverse Event Reporting System (FAERS) Essentials: A Guide to Understanding, Applying, and Interpreting Adverse Event Data | Potter, Reyes, Naples, Dal Pan | 2025 | Clinical Pharmacology & Therapeutics | 1 | Comprehensive methodological guide; 28M+ FAERS reports (20M+ unique after deduplication); critical caution: 40% of published FAERS studies misuse causal language; data cannot establish incidence rates or causality; disproportionality is hypothesis-generating only |
| Pharmacovigilance from social media: mining adverse drug reaction mentions using sequence labeling | Nikfarjam, Sarker, O'Connor, Ginn, Gonzalez | 2015 | Journal of the American Medical Informatics Association | 1 | Foundational NLP paper for pharmacovigilance text mining; CRF model for ADR extraction from Twitter; establishes the tradition of automated pharmacovigilance signal detection from unstructured text |

### Cluster 3 Synthesis

The FAERS database is widely used for drug-level signal detection (disproportionality analysis, PRR, ROR) but is rarely deployed as a *facility-level or manufacturer-level* outcome variable. Sardella et al. (2021) most explicitly frames pharmacovigilance as a manufacturing quality monitoring function — adverse event signals from contaminated sartan products, heparin adulteration, and formulation-change-induced immunogenicity all originated in manufacturing decisions, not pharmacological mechanisms. Brown (2020) provides the most direct precedent for our approach: he calls for systematic use of real-world data (FAERS, Sentinel) to generate facility-level quality signals, and notes that the FDA already folds FAERS hazard signals into its risk-based inspection site-selection model, creating an existing but informal loop between AE reports and facility-level regulatory attention. The Rahman et al. 2017 and Alatawi et al. 2017/2018 methodological papers establish that FAERS can distinguish manufacturing-origin quality differences across drug sources if perception bias is properly controlled. A critical limitation noted across Cluster 3: FAERS captures only reported adverse events, which reflect both pharmacological effects and manufacturing defects without distinguishing them, and reporting rates are heavily influenced by market share, media coverage, and patient/provider familiarity — all potential confounders when comparing facilities.

---

## Gaps / Our Contribution

The following capabilities or analyses do not appear in the existing literature:

| Gap | What the Literature Does | What We Do |
|-----|--------------------------|------------|
| NLP on FDA Form 483 observation narratives | Pazhayattil et al. 2020 counts 483s by frequency; no paper applies NLP/LLM to the raw text of 483 observations | Apply LLM extraction to 483 observation text to derive structured severity, scope, and CFR-domain features at the facility level |
| Connecting 483 text features to AE outcomes | Inspection-outcome studies (Wang et al. 2025; Stomberg 2017) use binary inspection classifications (OAI/VAI/NAI), not text-derived features | Use LLM-extracted text signals as continuous predictors of facility-level FAERS adverse event counts |
| Facility-level FAERS as outcome variable | Cluster 3 literature uses FAERS at drug-class or drug-manufacturer level; no paper uses facility-level FEI-aggregated FAERS counts as a quality outcome | Aggregate FAERS reports to the FEI (Facility Establishment Identifier) level and model them as a manufacturing quality outcome |
| Integrating inspection text + recall + warning letters + FAERS in one model | Each signal is studied in isolation across separate papers | Combine 483-text features, recall counts, warning-letter history, and inspection outcome into a unified facility-quality risk index linked to FAERS outcomes |
| Valisure ground-truth quality signal | No existing study uses independent third-party chemical testing results as ground-truth quality validation | Use Valisure independent lab testing results across 14 drugs as an external validation anchor for the quality risk index |

---

## Citation Seed List (Relevance = 3 Papers)

```bibtex
@article{Li2026LLMDevice,
  author    = {Hanyang Li and Xiao He and Adarsh Subbaswamy and Patrick Vossler and Alexej Gossmann and Karandeep Singh and Jean Feng},
  title     = {Scaling medical device regulatory science using large language models},
  journal   = {NPJ Digital Medicine},
  year      = {2026},
  doi       = {10.1038/s41746-026-02353-7},
  note      = {PDF: not available (paywall/login required)}
}

@article{Pazhayattil2020,
  author    = {Ajay Babu Pazhayattil and Marzena Ingram and Naheed Sayeed},
  title     = {A Quantitative Study of {US} {FDA} Inspection Data for Drug Manufacturing Sites},
  journal   = {Therapeutic Innovation \& Regulatory Science},
  year      = {2020},
  volume    = {54},
  pages     = {725--730},
  doi       = {10.1007/s43441-019-00015-3},
  note      = {PDF: not available (paywall); ResearchGate PDF available}
}

@article{Wang2025InspectionShortages,
  author    = {Yixin (Iris) Wang and George Ball and Gopesh Anand and Hyunwoo Park},
  title     = {Obligatory Responses to {FDA} Inspection Outcomes and Future Drug Shortages},
  journal   = {Manufacturing \& Service Operations Management},
  year      = {2025},
  volume    = {27},
  number    = {3},
  pages     = {789--807},
  doi       = {10.1287/msom.2022.0322},
  note      = {PDF: not available (INFORMS paywall)}
}

@techreport{Stomberg2017,
  author      = {Christopher Stomberg},
  title       = {Drug Shortages, Pricing, and Regulatory Activity},
  institution = {National Bureau of Economic Research},
  year        = {2017},
  number      = {W22912},
  note        = {Also published as chapter in: \emph{Measuring and Modeling Health Care Costs}, Univ. Chicago Press, 2018. PDF: not available (binary download)}
}

@article{Sardella2021,
  author    = {Marco Sardella and Glyn Belcher and Calin Lungu and {et al.}},
  title     = {Monitoring the manufacturing and quality of medicines: a fundamental task of pharmacovigilance},
  journal   = {Therapeutic Advances in Drug Safety},
  year      = {2021},
  doi       = {10.1177/20420986211038436},
  note      = {PDF: available via PMC (PMC8361554)}
}

@article{Brown2020GenericQuality,
  author    = {Joshua D. Brown},
  title     = {A Call to Action to Track Generic Drug Quality Using Real-World Data and the {FDA}'s Sentinel Initiative},
  journal   = {Journal of Managed Care \& Specialty Pharmacy},
  year      = {2020},
  volume    = {26},
  number    = {8},
  doi       = {10.18553/jmcp.2020.26.8.1050},
  note      = {PDF: available via PMC (PMC10390989)}
}

@article{Rahman2017FAERS,
  author    = {Md Motiur Rahman and Yasser Alatawi and Ning Cheng and Jingjing Qian and Peggy L. Peissig and Richard L. Berg and David C. Page and Richard A. Hansen},
  title     = {Methodological considerations for comparison of brand versus generic versus authorized generic adverse event reports in the {U.S.} Food and Drug Administration Adverse Event Reporting System ({FAERS})},
  journal   = {Clinical Drug Investigation},
  year      = {2017},
  doi       = {10.1007/s40261-017-0574-4},
  note      = {PDF: available via PMC (PMC5842081)}
}
```

---

## Full Paper Index by Cluster

### Cluster 1 Papers — Full Reference List

1. **Li et al. (2026)** — Scaling medical device regulatory science using LLMs. *NPJ Digital Medicine*. DOI: 10.1038/s41746-026-02353-7. Relevance: 3. PDF: not available.
2. **Pazhayattil et al. (2020)** — Quantitative study of US FDA inspection data. *Therapeutic Innovation & Regulatory Science*. DOI: 10.1007/s43441-019-00015-3. Relevance: 3. PDF: ResearchGate.
3. **Wu et al. (2021)** — BERT-Based NLP of drug labeling for DILI classification. *Frontiers in Artificial Intelligence*. DOI: 10.3389/frai.2021.729834. Relevance: 2. PDF: open access at PMC8685544.
4. **Bayer et al. (2021)** — ADE Eval: NLP evaluation for adverse event extraction from drug labels. *Drug Safety*. DOI: 10.1007/s40264-020-00996-3. Relevance: 2. PDF: not available (paywall).
5. **Tatonetti et al. (2025)** — OnSIDES: Extracting adverse drug events from drug labels. *Med (Cell Press)*. Relevance: 2. Preprint: medrxiv 10.1101/2024.03.22.24304724. PDF: preprint available.
6. **Rathore et al. (2022)** — FDA Warning Letters 2010-2020 retrospective analysis. *Journal of Pharmaceutical Innovation*. DOI: 10.1007/s12247-022-09678-2. Relevance: 2. PDF: available PMC9377664.
7. **Park & Kwon (2025)** — FDA data integrity enforcement: 1766 warning letters 2016-2023. *Therapeutic Innovation & Regulatory Science*. DOI: 10.1007/s43441-025-00870-3. Relevance: 2. PDF: not available.
8. **Kwiecinski (2024)** — Analysis of FDA warning letter citations 2019-2023. *Journal of Pharmaceutical Innovation*. DOI: 10.1007/s12247-024-09879-x. Relevance: 2. PDF: not available.
9. **Hassani et al. (2025)** — LLM-based classification of food-safety regulatory provisions. *arXiv:2501.14683*. Relevance: 2. PDF: open access arXiv.
10. **Atalay & Yigit-Sert (2026)** — RecallRisk-BERT for medical device recall triage. *arXiv:2606.27174*. Relevance: 2. PDF: open access arXiv.
11. **Barone et al. (2025)** — DART: Regulatory drug documents dataset for clinical NLP. *arXiv:2510.18475*. Relevance: 1. PDF: open access arXiv.

### Cluster 2 Papers — Full Reference List

12. **Wang et al. (2025)** — FDA Inspection Outcomes and Future Drug Shortages. *MSOM*. DOI: 10.1287/msom.2022.0322. Relevance: 3. PDF: not available (INFORMS paywall).
13. **Stomberg (2017)** — Drug Shortages, Pricing, and Regulatory Activity. *NBER W22912*. Relevance: 3. PDF: NBER website (binary; not extractable).
14. **Naumov et al. (2025)** — Evaluating quality reward interventions for drug shortages. *Journal of Operations Management*. DOI: 10.1002/joom.1334. Relevance: 2. PDF: not available (paywall).
15. **Lee et al. (2021)** — Alleviating drug shortages: mandated reporting transparency. *Management Science*. DOI: 10.1287/mnsc.2020.3857. Relevance: 2. PDF: not available (INFORMS paywall).
16. **Liu et al. (2021)** — Predictive model for drug shortages. *American Journal of Health-System Pharmacy*. DOI: 10.1093/ajhp/zxab152. Relevance: 2. PDF: available PMC8271205.
17. **Pall et al. (2023)** — Predicting drug shortages with pharmacy data and ML. *Health Care Management Science*. DOI: 10.1007/s10729-022-09627-y. Relevance: 2. PDF: available PMC10009839.
18. **Kosmas & Ergun (2023)** — When should FDA inspect manufacturing facilities? *arXiv:2310.15146*. Relevance: 2. PDF: open access arXiv.
19. **Slivinskis et al. (2024)** — ML algorithm to predict medical device recall by FDA. *Western Journal of Emergency Medicine*. DOI: 10.5811/westjem.21238. Relevance: 1. PDF: available PMC11908527.
20. **Shaban et al. (2018)** — Impact of drug shortages on patient safety and costs. *Federal Practitioner*. Relevance: 1. PDF: available PMC6248141.

### Cluster 3 Papers — Full Reference List

21. **Sardella et al. (2021)** — Monitoring manufacturing and quality: fundamental task of pharmacovigilance. *Therapeutic Advances in Drug Safety*. DOI: 10.1177/20420986211038436. Relevance: 3. PDF: available PMC8361554.
22. **Brown (2020)** — Call to action to track generic drug quality using real-world data. *JMCP*. DOI: 10.18553/jmcp.2020.26.8.1050. Relevance: 3. PDF: available PMC10390989.
23. **Rahman et al. (2017)** — Methodological considerations for brand vs. generic FAERS comparison. *Clinical Drug Investigation*. DOI: 10.1007/s40261-017-0574-4. Relevance: 3. PDF: available PMC5842081.
24. **Alatawi et al. (2017/2018)** — Brand vs. generic AE reporting: authorized generic-controlled evaluation. *Journal of Clinical Pharmacy and Therapeutics*. DOI: 10.1111/jcpt.12646. Relevance: 2. PDF: available PMC5930131.
25. **Rahman et al. (2017b)** — Comparison of brand vs. generic antiepileptic drug FAERS reporting rates. *Epilepsy Research*. DOI: 10.1016/j.eplepsyres.2017.06.007. Relevance: 2. PDF: available PMC5842137.
26. **Potter et al. (2025)** — FAERS Essentials: guide to understanding adverse event data. *Clinical Pharmacology & Therapeutics*. DOI: 10.1002/cpt.3701. Relevance: 1. PDF: available PMC12393772.
27. **Nikfarjam et al. (2015)** — Pharmacovigilance from social media: mining ADR mentions. *JAMIA*. DOI: 10.1093/jamia/ocu041. Relevance: 1. PDF: available PMC4457113.

---

## Notes on PDF Availability

### PDFs saved to `papers/` directory (confirmed valid):

| File | Source |
|------|--------|
| `2023_Kosmas_FDAInspectionTimingShortages.pdf` | arXiv:2310.15146 |
| `2025_Hassani_LLMFoodSafetyRegulations.pdf` | arXiv:2501.14683 |
| `2025_Barone_DART_RegulatoryNLP.pdf` | arXiv:2510.18475 |
| `2026_Atalay_RecallRiskBERT.pdf` | arXiv:2606.27174 |

### Open-access but not downloadable via script (PMC requires browser session):

Retrieve these manually by visiting the PMC URL in a browser:

| Paper | PMC ID | Direct URL |
|-------|--------|-----------|
| Sardella et al. 2021 | PMC8361554 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8361554/ |
| Brown 2020 | PMC10390989 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10390989/ |
| Rahman et al. 2017 (FAERS methodology) | PMC5842081 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5842081/ |
| Alatawi et al. 2017/2018 | PMC5930131 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5930131/ |
| Rahman et al. 2017b (AEDs) | PMC5842137 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5842137/ |
| Wu et al. 2021 (BERT DILI) | PMC8685544 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8685544/ |
| Liu et al. 2021 (drug shortage prediction) | PMC8271205 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8271205/ |
| Pall et al. 2023 (ML drug shortages) | PMC10009839 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10009839/ |
| Slivinskis et al. 2024 (device recall ML) | PMC11908527 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11908527/ |
| Shaban et al. 2018 (drug shortage patient safety) | PMC6248141 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6248141/ |
| Rathore et al. 2022 (warning letters) | PMC9377664 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9377664/ |

### Available via preprint server:

- Tatonetti et al. (OnSIDES) → https://www.medrxiv.org/content/10.1101/2024.03.22.24304724v1

### Paywalled (institutional access required):

- Wang et al. 2025 (MSOM): DOI 10.1287/msom.2022.0322
- Stomberg 2017 (NBER W22912): https://www.nber.org/papers/w22912
- Lee et al. 2021 (Management Science): DOI 10.1287/mnsc.2020.3857
- Naumov et al. 2025 (JOM): DOI 10.1002/joom.1334
- Bayer et al. 2021 (Drug Safety): DOI 10.1007/s40264-020-00996-3
- Park & Kwon 2025 (TIRS): DOI 10.1007/s43441-025-00870-3
- Kwiecinski 2024 (JPI): DOI 10.1007/s12247-024-09879-x
