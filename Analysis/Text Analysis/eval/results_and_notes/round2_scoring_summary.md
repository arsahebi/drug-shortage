# Round-2 human validation: scored results

Source: `eval/code/08_score_round2.py`. Claude Sonnet 5 v2 extraction against Abdel's blinded round-2 labels. Percent agreement with Wilson 95% intervals; n beside every figure. See the script docstring for conventions.

## PRIMARY: his own labels | new50

```
                                   field   n  agree   pct  ci_lo  ci_hi  kappa  pct_weighted  n_eff  ci_lo_weighted  ci_hi_weighted
                      violation_category  50     39 0.780  0.648  0.872  0.717         0.701 32.721           0.530           0.830
                           severity_tier  50     26 0.520  0.385  0.652  0.315         0.618 32.721           0.448           0.764
severity_tier (Major+Moderate collapsed)  50     32 0.640  0.501  0.759  0.184         0.779 32.721           0.612           0.888
                                   scope  50     34 0.680  0.542  0.792  0.530         0.661 32.721           0.490           0.799
                         root_cause_type  50     34 0.680  0.542  0.792  0.501         0.729 32.721           0.559           0.851
                      remediation_signal  50     40 0.800  0.670  0.888  0.254         0.877 32.721           0.724           0.951
                             repeat_flag  50     49 0.980  0.895  0.996  0.658         0.988 32.721           0.874           0.999
                       patient_risk_flag  50     39 0.780  0.648  0.872  0.212         0.794 32.721           0.628           0.898
                      contamination_flag  50     46 0.920  0.812  0.968  0.563         0.951 32.721           0.819           0.988
                 contamination_risk_flag  50     45 0.900  0.786  0.957  0.733         0.934 32.721           0.795           0.981
                      investigation_flag  50     43 0.860  0.738  0.930  0.687         0.815 32.721           0.651           0.912
                     data_integrity_flag  50     42 0.840  0.715  0.917  0.610         0.862 32.721           0.706           0.942
                      POOLED (11 fields) 550    437 0.795  0.747  0.838    NaN         0.812 32.721           0.761           0.858
```

## PRIMARY: his own labels | all70

```
                                   field   n  agree   pct  ci_lo  ci_hi  kappa
                      violation_category  70     55 0.786  0.676  0.866  0.725
                           severity_tier  70     42 0.600  0.483  0.707  0.395
severity_tier (Major+Moderate collapsed)  70     50 0.714  0.599  0.807  0.226
                                   scope  70     48 0.686  0.570  0.782  0.550
                         root_cause_type  70     48 0.686  0.570  0.782  0.499
                      remediation_signal  70     58 0.829  0.724  0.899  0.225
                             repeat_flag  70     67 0.957  0.881  0.985  0.386
                       patient_risk_flag  70     56 0.800  0.692  0.877  0.230
                      contamination_flag  70     62 0.886  0.790  0.941  0.371
                 contamination_risk_flag  70     59 0.843  0.740  0.910  0.612
                      investigation_flag  70     61 0.871  0.773  0.931  0.715
                     data_integrity_flag  70     59 0.843  0.740  0.910  0.574
                      POOLED (11 fields) 770    615 0.799  0.760  0.836    NaN
```

## PRIMARY: his own labels | random_natural20

```
                                   field   n  agree   pct  ci_lo  ci_hi  kappa
                      violation_category  20     14 0.700  0.481  0.855  0.562
                           severity_tier  20     11 0.550  0.342  0.742  0.331
severity_tier (Major+Moderate collapsed)  20     14 0.700  0.481  0.855  0.178
                                   scope  20     13 0.650  0.433  0.819  0.417
                         root_cause_type  20     15 0.750  0.531  0.888  0.587
                      remediation_signal  20     16 0.800  0.584  0.919  0.000
                             repeat_flag  20     19 0.950  0.764  0.991  0.643
                       patient_risk_flag  20     14 0.700  0.481  0.855  0.178
                      contamination_flag  20     18 0.900  0.699  0.972  0.615
                 contamination_risk_flag  20     20 1.000  0.839  1.000  1.000
                      investigation_flag  20     17 0.850  0.640  0.948  0.700
                     data_integrity_flag  20     17 0.850  0.640  0.948  0.667
                      POOLED (11 fields) 220    174 0.791  0.723  0.855    NaN
```

## SENSITIVITY ONLY: after his Claude self-check | new50

```
                                   field   n  agree   pct  ci_lo  ci_hi  kappa  pct_weighted  n_eff  ci_lo_weighted  ci_hi_weighted
                      violation_category  50     39 0.780  0.648  0.872  0.718         0.701 32.721           0.530           0.830
                           severity_tier  50     27 0.540  0.404  0.670  0.339         0.655 32.721           0.484           0.794
severity_tier (Major+Moderate collapsed)  50     33 0.660  0.522  0.776  0.240         0.817 32.721           0.653           0.913
                                   scope  50     34 0.680  0.542  0.792  0.531         0.661 32.721           0.490           0.799
                         root_cause_type  50     34 0.680  0.542  0.792  0.501         0.729 32.721           0.559           0.851
                      remediation_signal  50     40 0.800  0.670  0.888  0.254         0.877 32.721           0.724           0.951
                             repeat_flag  50     49 0.980  0.895  0.996  0.658         0.988 32.721           0.874           0.999
                       patient_risk_flag  50     40 0.800  0.670  0.888  0.307         0.831 32.721           0.670           0.923
                      contamination_flag  50     46 0.920  0.812  0.968  0.563         0.951 32.721           0.819           0.988
                 contamination_risk_flag  50     45 0.900  0.786  0.957  0.733         0.930 32.721           0.791           0.979
                      investigation_flag  50     43 0.860  0.738  0.930  0.687         0.815 32.721           0.651           0.912
                     data_integrity_flag  50     43 0.860  0.738  0.930  0.664         0.875 32.721           0.721           0.950
                      POOLED (11 fields) 550    440 0.800  0.751  0.844    NaN         0.819 32.721           0.766           0.865
```

## Test-retest: round 2 vs round 1, 20 repeats (Abdel against himself)

```
                                   field   n  agree   pct  ci_lo  ci_hi  kappa rule_amended_since_round1
                      violation_category  20     17 0.850  0.640  0.948  0.803                     False
                           severity_tier  20     16 0.800  0.584  0.919  0.658                     False
severity_tier (Major+Moderate collapsed)  20     17 0.850  0.640  0.948  0.348                     False
                                   scope  20     16 0.800  0.584  0.919  0.724                     False
                         root_cause_type  20     15 0.750  0.531  0.888  0.618                     False
                      remediation_signal  20     20 1.000  0.839  1.000    NaN                     False
                             repeat_flag  20     18 0.900  0.699  0.972  0.000                     False
                       patient_risk_flag  20     17 0.850  0.640  0.948  0.500                      True
                      contamination_flag  20     16 0.800  0.584  0.919  0.412                      True
                 contamination_risk_flag  20     18 0.900  0.699  0.972  0.733                      True
                      investigation_flag  20     19 0.950  0.764  0.991  0.886                     False
                     data_integrity_flag  20     18 0.900  0.699  0.972  0.459                      True
                      POOLED (11 fields) 220    190 0.864  0.795  0.918    NaN                       NaN
```

## Direction of the logged AI edits

```
direction
to Claude           18
free-text field      9
neither              4
away from Claude     2
```

Categorical and flag fields only:

```
direction
to Claude           18
neither              4
away from Claude     2
```
