# Text-signal AE prediction — model summary

Panel: 246 inspection events, 98 unique FEIs
Outcome: above-median AEs in Q+1..Q+4 after inspection (base rate 50.0%)

            config model      auc       ap
      A: Text only    LR 0.528052 0.546468
      A: Text only    RF 0.502056 0.512689
     B: Text + OAI    LR 0.521235 0.540889
     B: Text + OAI    RF 0.495662 0.512627
  C: OAI flag only    LR 0.463480 0.491409
  C: OAI flag only    RF 0.463480 0.491409
D: VAI-only (text)    LR 0.495704 0.524738
D: VAI-only (text)    RF 0.437815 0.490846
E: OAI-ever (text)    LR 0.596032 0.700991
E: OAI-ever (text)    RF 0.433247 0.576103

AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.

## AE trajectory by facility group
          group  n_feis  n_rows  mean_ae_tm4  mean_ae_tm2  mean_ae_t0  mean_ae_tp2  mean_ae_tp4  pre_rise_t0_tm4  persist_tp4_t0
       OAI-ever      25      81        567.2        617.2       643.9        681.9        661.6            1.135           1.027
High-signal VAI      14      26        661.1        781.7       796.6        886.5        868.8            1.205           1.091
 Low-signal VAI      59     139        468.9        524.7       547.7        582.8        578.7            1.168           1.057