# VAI-signal validation rerun -- model summary

Outcome definition: relative (ae_rise_next4q)
Panel: 119 inspection events, 62 unique FEIs
Outcome base rate: 55.5%

                    config model      auc  p_vs_0.5       ap
A: Text only (full sample)    LR 0.564325  0.118255 0.626561
A: Text only (full sample)    RF 0.639274  0.009240 0.695084
             B: Text + OAI    LR 0.550305  0.201810 0.635520
             B: Text + OAI    RF 0.617969  0.009301 0.700204
          C: OAI flag only    LR 0.537621  0.224315 0.578670
          C: OAI flag only    RF 0.537621  0.224315 0.578670
        D: VAI-only (text)    LR 0.456270  0.740844 0.713965
        D: VAI-only (text)    RF 0.515888  0.428543 0.730626
        E: OAI-ever (text)    LR 0.636810  0.040735 0.647460
        E: OAI-ever (text)    RF 0.727333  0.000164 0.708532

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.
p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching
the INFORMS slide's own stated test.