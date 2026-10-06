# VAI-signal validation rerun -- model summary

Outcome definition: global (ae_high_next4q)
Panel: 176 inspection events, 78 unique FEIs
Outcome base rate: 50.0%

                    config model      auc  p_vs_0.5       ap
A: Text only (full sample)    LR 0.483016  0.610562 0.531554
A: Text only (full sample)    RF 0.531788  0.230667 0.592354
             B: Text + OAI    LR 0.488448  0.582613 0.519600
             B: Text + OAI    RF 0.534378  0.250590 0.596197
          C: OAI flag only    LR 0.537217  0.168667 0.526775
          C: OAI flag only    RF 0.537217  0.168667 0.526775
        D: VAI-only (text)    LR 0.590658  0.188853 0.583072
        D: VAI-only (text)    RF 0.610054  0.079600 0.570596
        E: OAI-ever (text)    LR 0.618983  0.078663 0.740353
        E: OAI-ever (text)    RF 0.337684  0.986750 0.594528

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.
p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching
the INFORMS slide's own stated test.