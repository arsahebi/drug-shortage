# VAI-signal validation rerun -- model summary

Outcome definition: relative (ae_rise_next4q)
Panel: 238 inspection events, 98 unique FEIs
Outcome base rate: 67.2%

                    config model      auc  p_vs_0.5       ap
A: Text only (full sample)    LR 0.513996  0.282312 0.689001
A: Text only (full sample)    RF 0.549476  0.137806 0.704634
             B: Text + OAI    LR 0.512667  0.289734 0.686091
             B: Text + OAI    RF 0.558162  0.137601 0.713414
          C: OAI flag only    LR 0.520827  0.029853 0.682097
          C: OAI flag only    RF 0.520827  0.029853 0.682097
        D: VAI-only (text)    LR 0.460653  0.723688 0.687860
        D: VAI-only (text)    RF 0.467317  0.682438 0.700902
        E: OAI-ever (text)    LR 0.500909  0.494377 0.701688
        E: OAI-ever (text)    RF 0.566970  0.173508 0.734541

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.
p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching
the INFORMS slide's own stated test.