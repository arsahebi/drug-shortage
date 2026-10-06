# VAI-signal validation rerun -- model summary

Outcome definition: global (ae_high_next4q)
Panel: 238 inspection events, 98 unique FEIs
Outcome base rate: 50.0%

                    config model      auc  p_vs_0.5       ap
A: Text only (full sample)    LR 0.571688  0.034659 0.566094
A: Text only (full sample)    RF 0.544859  0.062302 0.555042
             B: Text + OAI    LR 0.564705  0.036127 0.561430
             B: Text + OAI    RF 0.540556  0.074518 0.563205
          C: OAI flag only    LR 0.510542  0.215393 0.505425
          C: OAI flag only    RF 0.510542  0.215393 0.505425
        D: VAI-only (text)    LR 0.577324  0.001158 0.582546
        D: VAI-only (text)    RF 0.583267  0.014879 0.590851
        E: OAI-ever (text)    LR 0.507961  0.432259 0.612163
        E: OAI-ever (text)    RF 0.583748  0.038453 0.764923

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.
p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching
the INFORMS slide's own stated test.