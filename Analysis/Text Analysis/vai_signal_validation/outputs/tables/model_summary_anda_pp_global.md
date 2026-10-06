# VAI-signal validation rerun -- model summary

Outcome definition: global (ae_high_next4q)
Panel: 164 inspection events, 73 unique FEIs
Outcome base rate: 49.4%

                    config model      auc  p_vs_0.5       ap
A: Text only (full sample)    LR 0.595569  0.015616 0.611054
A: Text only (full sample)    RF 0.586548  0.005615 0.605304
             B: Text + OAI    LR 0.584061  0.029673 0.604842
             B: Text + OAI    RF 0.585000  0.008437 0.605710
          C: OAI flag only    LR 0.507540  0.368433 0.502604
          C: OAI flag only    RF 0.507540  0.368433 0.502604
        D: VAI-only (text)    LR 0.610879  0.024996 0.557210
        D: VAI-only (text)    RF 0.625589  0.002524 0.568937
        E: OAI-ever (text)    LR 0.640506  0.072896 0.692843
        E: OAI-ever (text)    RF 0.557569  0.186945 0.619016

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.
p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching
the INFORMS slide's own stated test.