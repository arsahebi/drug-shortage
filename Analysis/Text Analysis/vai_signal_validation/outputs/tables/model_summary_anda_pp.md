# VAI-signal validation rerun -- model summary

Outcome definition: relative (ae_rise_next4q)
Panel: 143 inspection events, 65 unique FEIs
Outcome base rate: 55.9%

                    config model      auc  p_vs_0.5       ap
A: Text only (full sample)    LR 0.566698  0.072885 0.630715
A: Text only (full sample)    RF 0.648383  0.031617 0.712947
             B: Text + OAI    LR 0.564672  0.070736 0.627055
             B: Text + OAI    RF 0.650927  0.030703 0.715954
          C: OAI flag only    LR 0.521074  0.258530 0.571618
          C: OAI flag only    RF 0.521074  0.258530 0.571618
        D: VAI-only (text)    LR 0.447186  0.643129 0.691667
        D: VAI-only (text)    RF 0.508398  0.471174 0.697696
        E: OAI-ever (text)    LR 0.630212  0.074807 0.677475
        E: OAI-ever (text)    RF 0.740899  0.040849 0.768987

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.
p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching
the INFORMS slide's own stated test.