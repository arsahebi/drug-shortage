"""Use the existing 17 text signals against separate DoD scoring components.

No keyword exposures or route/product screens are constructed. The existing
step02 inspection features are read verbatim. Source component signs, absent
columns and zero recorded penalties are preserved and independently reconciled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("valisure_source_helpers", HERE/"20261006_validate_text_against_valisure.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
FEATURE_FILE = base.TEXT/"step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
DEFINITION_FILE = base.TEXT/"vai_signal_validation/02_vai_signal_model.py"
OUTCOMES = ["DoD_score", "D_dissolution", "DMF", "NDMA_nitrosamines", "T_toxic"]


def fixed_features():
    tree = ast.parse(DEFINITION_FILE.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "TEXT_FEATURES" for t in node.targets):
            features = ast.literal_eval(node.value)
            assert len(features) == len(set(features)) == 17
            return features
    raise ValueError("Existing TEXT_FEATURES definition not found")


def split_components(scores):
    rows = []
    for _,r in scores.iterrows():
        vals = {k:0.0 if v == "--" else float(v) for k,v in json.loads(r.component_raw).items()}
        def amount(tokens):
            items = [v for k,v in vals.items() if any(t in k.lower() for t in tokens)]
            return -sum(items) if items else np.nan
        rec = {"score_card_id":r.score_card_id, "DoD_score":r.score,
               "D_dissolution":amount(["dissolution"]), "DMF":amount(["dmf"]),
               "NDMA_nitrosamines":amount(["nitrosamin","ndma"]),
               "T_toxic":amount(["toxic element"]), "dosage":amount(["dosage"]),
               "benzene_etox":amount(["benzene","etox"]), "sterility":amount(["sterility"])}
        for key in rec.keys()-{"score_card_id","DoD_score"}:
            rec[key+"_signed_source"] = -rec[key]
        assert np.isclose(rec["DoD_score"],100-sum(rec[k] for k in ["D_dissolution","DMF","NDMA_nitrosamines","T_toxic","dosage","benzene_etox","sterility"] if pd.notna(rec[k])))
        rows.append(rec)
    return scores.merge(pd.DataFrame(rows),on="score_card_id",validate="one_to_one")


def attach_fixed_text(s,features):
    ts = pd.read_csv(FEATURE_FILE,dtype={"fei":str})
    ts["fei"] = ts.fei.map(base.fei_key)
    ts["snapshot_date"] = pd.to_datetime(ts.snapshot_date)
    assert not ts.duplicated(["fei","snapshot_date"]).any()
    for col in features:
        ts[col] = pd.to_numeric(ts[col],errors="raise").astype(float)
    dates = pd.read_excel(base.FILES["intake"],sheet_name="2024 Testing Data",header=1)
    dates = dates[dates["Sample ID"].notna()].copy()
    dates["intake_date"] = pd.to_datetime(dates["Intake Date"].astype(str).str.replace(r"\.0$","",regex=True),format="%Y%m%d",errors="coerce")
    assert not dates["Sample ID"].duplicated().any()
    s = s.copy()
    s["intake_date"] = s.sample_id.map(dates.set_index("Sample ID").intake_date)
    s["cutoff_date"] = s.intake_date.fillna(pd.Timestamp("2023-01-01"))
    s["date_basis"] = np.where(s.intake_date.notna(),"observed_metformin_intake","conservative_2023_boundary")
    g = pd.read_parquet(base.FILES["grades"],columns=["fei","insp_date","n_oai","n_vai","n_nai"])
    g["fei"],g["insp_date"] = g.fei.map(base.fei_key),pd.to_datetime(g.insp_date)
    grade = {}
    for key,x in g.groupby(["fei","insp_date"]):
        grade[key] = "OAI" if x.n_oai.max()>0 else ("VAI" if x.n_vai.max()>0 else ("NAI" if x.n_nai.max()>0 else "UNKNOWN"))
    groups = {f:x.sort_values("snapshot_date") for f,x in ts.groupby("fei")}
    records = []
    for _,r in s.iterrows():
        rec = {"sample_id":r.sample_id,"text_match_status":"no_unique_facility"}
        x = groups.get(r.fei)
        if r.fei and (x is None or not (x.snapshot_date<r.cutoff_date).any()):
            rec["text_match_status"] = "no_prior_text"
        elif r.fei:
            row = x[x.snapshot_date<r.cutoff_date].iloc[-1]
            rec.update({c:row[c] for c in features})
            rec.update(text_match_status="matched",inspection_date=row.snapshot_date,
                       inspection_age_days=int((r.cutoff_date-row.snapshot_date).days),
                       n_obs_compatible=row.n_obs_total,n_obs_total=row.n_obs_total,
                       fda_class=grade.get((r.fei,row.snapshot_date),"UNKNOWN"))
        records.append(rec)
    return s.merge(pd.DataFrame(records),on="sample_id",validate="one_to_one")


def make_cells(s,features):
    x = s[s.text_match_status.eq("matched")&s.score_match_status.eq("matched")].copy()
    x["cluster"] = base.connected_groups(x)
    keys = ["fei","product","api","form","strength","score_unit","cluster"]
    nums = features+OUTCOMES+["inspection_age_days","n_obs_compatible","n_obs_total"]
    spec = {c:(c,"mean") for c in nums}
    spec.update(n_samples=("sample_id","nunique"),sample_ids=("sample_id",lambda a:" ; ".join(sorted(set(a)))),
                fda_class=("fda_class",lambda a:next(iter(set(a))) if len(set(a))==1 else "MIXED"))
    d = x.groupby(keys,dropna=False).agg(**spec).reset_index()
    d["n_product_facilities"] = d.groupby("product").fei.transform("nunique")
    d["analysis_weight"] = 1/d.groupby("score_unit").fei.transform("size")
    return d


def endpoint_frame(d,target):
    x = d.dropna(subset=[target]).copy()
    x = x[x.groupby("product").fei.transform("nunique").ge(2)].copy().reset_index(drop=True)
    x["analysis_weight"] = 1/x.groupby("score_unit").fei.transform("size")
    return x


def continuous_prediction(d,target,features,cohort,n_boot):
    x = endpoint_frame(d,target)
    common = {"cohort":cohort,"outcome":target,"n_cells":len(x),"n_facilities":x.fei.nunique(),"n_clusters":x.cluster.nunique()}
    if x[target].nunique()<2:
        return [{**common,"status":"constant_outcome"}],pd.DataFrame()
    if len(x)<30 or x.cluster.nunique()<8:
        return [{**common,"status":"too_few_independent_observations"}],pd.DataFrame()
    x["log_age"],x["log_n_obs"] = np.log1p(x.inspection_age_days),np.log1p(x.n_obs_total)
    y,w,groups = x[target].to_numpy(),x.analysis_weight.to_numpy(),x.cluster.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x,y,groups))
    rows,pred_rows = [],[]
    for model_name in ["Ridge","RF"]:
        prediction = {"baseline":np.full(len(x),np.nan),"baseline_plus_fixed_text":np.full(len(x),np.nan)}
        fold_numbers = np.zeros(len(x),dtype=int)
        for i,(train,test) in enumerate(folds,1):
            fold_numbers[test] = i
            for name,cols in [("baseline",["log_age","log_n_obs"]),("baseline_plus_fixed_text",["log_age","log_n_obs"]+features)]:
                prep = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),["product","fda_class"]),("num",StandardScaler(),cols)])
                estimator = Ridge(alpha=10,solver="lsqr") if model_name=="Ridge" else RandomForestRegressor(n_estimators=300,max_depth=4,min_samples_leaf=5,random_state=42,n_jobs=1)
                model = make_pipeline(prep,estimator)
                parameter = "ridge__sample_weight" if model_name=="Ridge" else "randomforestregressor__sample_weight"
                model.fit(x.iloc[train],y[train],**{parameter:w[train]})
                prediction[name][test] = model.predict(x.iloc[test])
        metrics = {}
        for name,p in prediction.items():
            metrics[name+"_rmse"] = np.sqrt(mean_squared_error(y,p,sample_weight=w))
            metrics[name+"_mae"] = mean_absolute_error(y,p,sample_weight=w)
            metrics[name+"_r2"] = r2_score(y,p,sample_weight=w)
        unique = np.unique(groups)
        positions = {g:np.flatnonzero(groups==g) for g in unique}
        rng = np.random.default_rng(20261006)
        deltas = []
        for _ in range(n_boot):
            idx = np.concatenate([positions[g] for g in rng.choice(unique,len(unique),replace=True)])
            rmse = [np.sqrt(np.average((y[idx]-prediction[n][idx])**2,weights=w[idx])) for n in ["baseline","baseline_plus_fixed_text"]]
            deltas.append(rmse[1]-rmse[0])
        interval = np.quantile(deltas,[.025,.975])
        rows.append({**common,"status":"estimated","model":model_name,**metrics,
                     "rmse_difference":metrics["baseline_plus_fixed_text_rmse"]-metrics["baseline_rmse"],
                     "difference_ci_low":interval[0],"difference_ci_high":interval[1],
                     "uncertainty":"Paired connected-group bootstrap of fixed OOF predictions; conditional on training"})
        pred = x[["fei","product","score_unit","cluster","analysis_weight"]].copy()
        pred["cohort"],pred["outcome"],pred["model"],pred["observed"],pred["fold"] = cohort,target,model_name,y,fold_numbers
        for name,p in prediction.items():pred[name] = p
        assert pred.groupby("cluster").fold.nunique().max()==1
        pred_rows.append(pred)
    return rows,pd.concat(pred_rows,ignore_index=True)


def report(out,features,scores,flow,availability,associations,prediction):
    main = associations[associations.cohort.eq("all_source_unique")]
    hits = main[main.p_holm.lt(.05)]
    text = f"""# Fixed extracted text signals versus DoD scoring components

This revised analysis uses the same **17 existing text variables** as the FAERS/MarketScan models, read directly from the Claude v2 step02 feature file. It introduces no keyword variables, no new extraction, and no route/product filtering of observations. It supersedes the keyword-screen first pass, preserved in `../20261006_first_pass/`.

## Outcomes and scoring

The scoring workbook contains 302 rows across 13 API worksheets, including separate ampicillin–sulbactam labels. Every scoring row has an NDC represented in the testing workbook. Each original score reconciles exactly to:

`DoD score = 100 + sum(signed source components) = 100 - sum(positive penalty amounts)`.

Source entries such as -10, -30 and -61 are preserved. `--` is zero **recorded scoring penalty**, not a measured zero concentration. A component column absent from an API worksheet is missing, not zero. Scores below zero are retained. The score reconstruction includes dosage, benzene/EtOx and sterility where present, even though the five requested main outcomes are the total score, dissolution, DMF, nitrosamines and toxic elements.

The four component outcomes are positive penalty magnitudes: higher is worse. The overall outcome is the original score: higher is better. Coefficient signs therefore have opposite interpretations for total score versus penalties. The source labels the nitrosamine category as “Nitrosamines,” not an NDMA-specific concentration.

{base.markdown_table(availability,list(availability.columns))}

Nitrosamine penalties are zero in all 302 scoring rows. Their relationship to text cannot be estimated from these files. Nonzero DMF penalties occur only in the metformin worksheet, and the main analysis includes just four connected clusters with a DMF penalty. Toxic-element penalties sum the source's toxic-element columns where several metals are scored separately; original columns remain in `component_raw` and `score_components.csv`. Raw laboratory dissolution values are retained in the sample file but are not dependent variables in this revision.

## Cohort

{base.markdown_table(flow,list(flow.columns))}

Facility mapping and time safeguards are carried forward from the first pass. Main linkage requires a singleton in the complete cross-source map before selecting text-covered plants. Latest features must precede known metformin intake dates; otherwise the cutoff is January 1, 2023, based on the documented collection period. This is a provisional facility-level quality association, not proven lot-specific attribution or patient harm.

## Existing fixed variables

""" + "\n".join(f"- `{f}`" for f in features) + f"""

These definitions were not changed. The existing `joint_contamination_labcontrols` label is retained, but its implementation represents laboratory/facilities-system co-occurrence; it should not be reinterpreted as a newly confirmed contamination mechanism.

## Individual associations

Each of the 17 variables is tested separately against each of the five outcomes, adjusting within exact API/form/strength and for inspection age/observation count. Shared scoring units receive one total weight, and connected plant/score-unit clusters determine uncertainty. Effects are outcome points per one SD of the existing feature. No outcome-specific selection of text variables is performed.

All 85 feature/outcome combinations per cohort are reported in `fixed_signal_associations.csv`, including constant and unestimable cases. Holm correction covers the complete 85-test family for each cohort. Models with very sparse positive exposure remain descriptive. These exploratory models reuse previously examined data and are not confirmatory tests.

Main-cohort combinations surviving Holm correction: **{len(hits)}**.

{base.markdown_table(main.sort_values('p_value',na_position='last').head(12),['outcome','feature','status','n_facilities','effect_per_sd','ci_low','ci_high','p_value','p_holm'])}

The table shows the smallest nominal p-values for navigation; the full matrix includes every specified combination. Unadjusted findings should not be selected as a new primary hypothesis in this same dataset.

## Joint prediction with all 17 fixed signals

The outcomes retain their numeric values. Ridge regression (fixed alpha 10) and a constrained random forest (300 trees, depth 4, minimum leaf 5) compare the same baseline with and without all 17 existing signals. The baseline contains product, FDA classification, inspection age and observation count. Identical connected groups stay together in each five-fold comparison. Models are not tuned to select a winning endpoint or feature subset.

{base.markdown_table(prediction[prediction.cohort.eq('all_source_unique')],['outcome','model','status','n_cells','n_facilities','baseline_rmse','baseline_plus_fixed_text_rmse','rmse_difference','difference_ci_low','difference_ci_high','baseline_r2','baseline_plus_fixed_text_r2'])}

Lower RMSE is better. A negative difference means text improved prediction. Paired intervals bootstrap fixed out-of-fold predictions by connected group; they do not include all training/specification uncertainty. Negative R² means performance worse than the evaluated cohort's weighted overall-mean benchmark. All model predictions and sensitivities are exported.

The fixed signals do not show a reliable predictive gain in this dataset. For the total score, the random forest's RMSE falls from 26.64 to 25.60, but its paired interval includes no improvement; Ridge worsens from 24.97 to 27.26. Neither model improves the component outcomes reliably. Alongside the lack of multiplicity-adjusted associations, these results support reporting this as an exploratory laboratory-quality validation with limited evidence, rather than as demonstrated patient-outcome validation.

## Files and limits

`score_components.csv` preserves individual scoring columns and signs. `samples_fixed_all_source_unique.csv` preserves every sample, source score/component, linkage status and selected existing inspection features. `fixed_signal_cells_all_source_unique.csv` is the analysis panel. `fixed_signal_associations.csv` is the full association matrix. `fixed_signal_prediction.csv` and `fixed_signal_oof_predictions.csv` give paired continuous-outcome prediction results. `fixed_signal_manifest.json` records input hashes and settings.

Repeated NDCs/samples are not independent score measurements. Manufacturing links remain historical/role-sensitive. Dates are partly assumed collection boundaries. Generic facility signals need not concern the tested product. None of these results establishes patient risk or clinical harm.
"""
    (out/"RESULTS.md").write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=HERE/"outputs/20261006")
    p.add_argument("--bootstraps",type=int,default=2000)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    features = fixed_features()
    samples,assays = base.read_samples()
    scores,expanded = base.read_scores()
    scores = split_components(scores)
    expanded = expanded.merge(scores[["score_card_id"]+OUTCOMES],on="score_card_id",validate="many_to_one")
    samples = base.attach_scores(samples,expanded)
    unit_values = scores.groupby("score_unit")[OUTCOMES].first()
    for target in OUTCOMES:samples[target] = samples.score_unit.map(unit_values[target])
    audit,union,march = base.facility_links(set(samples.ndc9))
    scores.to_csv(out/"score_components.csv",index=False)
    audit.to_csv(out/"fixed_signal_facility_links.csv",index=False)
    availability = []
    for target in OUTCOMES:
        v = scores[target].dropna()
        availability.append({"outcome":target,"source_rows_present":len(v),"source_rows_with_penalty":int(v.gt(0).sum()) if target!='DoD_score' else int(v.lt(100).sum()),"distinct_values":v.nunique(),"minimum":v.min(),"maximum":v.max()})
    availability = pd.DataFrame(availability)
    availability.to_csv(out/"component_availability.csv",index=False)
    result,flows,metrics,predictions = [],[],[],[]
    for cohort,mapping in [("all_source_unique",union),("march_only_unique",march),("dated_metformin",union)]:
        s = samples.copy()
        s["candidate_feis"] = s.ndc9.map(lambda n:" ; ".join(union.get(n,[])))
        s["fei"] = s.ndc9.map(lambda n:mapping[n][0] if len(mapping.get(n,[]))==1 else "")
        s = attach_fixed_text(s,features)
        if cohort=="dated_metformin":s = s[s.intake_date.notna()].copy()
        d = make_cells(s,features)
        s.to_csv(out/f"samples_fixed_{cohort}.csv",index=False)
        d.to_csv(out/f"fixed_signal_cells_{cohort}.csv",index=False)
        m = s[s.text_match_status.eq("matched")]
        flows.append({"cohort":cohort,"source_samples":len(s),"prior_text_samples":len(m),"text_facilities":m.fei.nunique(),"score_cells":len(d),"shared_score_units":d.score_unit.nunique()})
        for target in OUTCOMES:
            for feature in features:
                a = base.associate(d,target,feature,f"{feature} versus {target}",cohort)
                if d[target].dropna().nunique()<=1:a.update(status="constant_outcome",reason="No outcome variation")
                result.append(a)
            rows,pred = continuous_prediction(d,target,features,cohort,args.bootstraps)
            metrics.extend(rows)
            if not pred.empty:predictions.append(pred)
    result = pd.DataFrame(result)
    result["p_holm"] = np.nan
    for _,g in result.groupby("cohort"):
        v = g[g.p_value.notna()]
        if len(v):
            family = np.r_[v.p_value,np.ones(len(OUTCOMES)*len(features)-len(v))]
            result.loc[v.index,"p_holm"] = multipletests(family,method="holm")[1][:len(v)]
    metrics,flow = pd.DataFrame(metrics),pd.DataFrame(flows)
    result.to_csv(out/"fixed_signal_associations.csv",index=False)
    metrics.to_csv(out/"fixed_signal_prediction.csv",index=False)
    flow.to_csv(out/"fixed_signal_cohort_flow.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/"fixed_signal_oof_predictions.csv",index=False)
    inputs = {k:v for k,v in base.FILES.items() if k!='observations'}
    inputs.update(fixed_features=FEATURE_FILE,feature_definition=DEFINITION_FILE,loader=Path(base.__file__))
    manifest = {k:{"path":str(v.relative_to(base.ROOT)),"sha256":hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in inputs.items()}
    manifest["run"] = {"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"features":features,"outcomes":OUTCOMES,"bootstraps":args.bootstraps,"seed":20261006,"ridge_alpha":10}
    (out/"fixed_signal_manifest.json").write_text(json.dumps(manifest,indent=2))
    report(out,features,scores,flow,availability,result,metrics)
    print(flow.to_string(index=False))
    print(availability.to_string(index=False))
    print(result[result.cohort.eq("all_source_unique")].sort_values("p_value").head(12)[["outcome","feature","effect_per_sd","p_value","p_holm"]].to_string(index=False))
    print(metrics[metrics.cohort.eq("all_source_unique")].to_string(index=False))
    print(f"Updated {out/'RESULTS.md'}")


if __name__=="__main__":
    main()
