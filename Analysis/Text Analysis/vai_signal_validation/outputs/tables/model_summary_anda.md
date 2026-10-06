# VAI-signal validation rerun -- model summary

Outcome definition: relative (ae_rise_next4q)
Panel: 156 inspection events, 66 unique FEIs
Outcome base rate: 59.6%

                    config model      auc  p_vs_0.5       ap
A: Text only (full sample)    LR 0.519898  0.305833 0.657353
A: Text only (full sample)    RF 0.625648  0.015294 0.715603
             B: Text + OAI    LR 0.508657  0.400430 0.651438
             B: Text + OAI    RF 0.608529  0.027851 0.705079
          C: OAI flag only    LR 0.456990  0.985034 0.579059
          C: OAI flag only    RF 0.456990  0.985034 0.579059
        D: VAI-only (text)    LR 0.533241  0.211951 0.696892
        D: VAI-only (text)    RF 0.501659  0.477740 0.633308
        E: OAI-ever (text)    LR 0.465584  0.681717 0.665741
        E: OAI-ever (text)    RF 0.570346  0.243934 0.714100

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.
p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching
the INFORMS slide's own stated test.