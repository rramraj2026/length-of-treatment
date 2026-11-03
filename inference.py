import numpy as np
import pandas as pd


def add_derived(X):
    X = X.copy()
    X['severity_tier'] = np.where((X.whz < -3) | (X.muac < 11.5), 2,
                         np.where((X.whz < -2) | (X.muac < 12.5), 1, 0)).astype(float)
    X['muac_deficit'] = np.clip(12.5 - X.muac, 0, None)
    X['whz_deficit'] = np.clip(-2 - X.whz, 0, None)
    X['stunted'] = (X.haz < -2).astype(float)
    X['severe_both'] = ((X.whz < -3) & (X.muac < 11.5)).astype(float)
    X['muac_x_whz'] = X.muac * X.whz
    X['haz_x_muac'] = X.haz * X.muac
    X['caregiver_age_x_edu'] = X.caregiver_age * X.maternal_edu
    X['vulnerability_index'] = X.ipc_phase + X.hunger_scale + (1 - X.handwashing)
    X['vulnerability_no_hunger'] = X.ipc_phase + (1 - X.handwashing)
    return X


def assemble(T, Xbase, fac_k, reg_k, sam_k, cluster_cols):
    out = Xbase.copy()
    fa, ra = np.zeros(len(out)), np.zeros(len(out))
    for v in (0., 1.):
        mk = sam_k == v
        if mk.any():
            tbl, pop = T[('fac_avg_duration', v)]
            fa[mk] = [tbl.get(k, pop) for k in fac_k[mk]]
            tbl, pop = T[('reg_avg_duration', v)]
            ra[mk] = [tbl.get(k, pop) for k in reg_k[mk]]
    out['fac_avg_duration'] = fa
    out['reg_avg_duration'] = ra
    tbl, pop = T['fac_relapse_rate']
    out['fac_relapse_rate'] = [tbl.get(k, pop) for k in fac_k]
    tbl, pop = T['fac_recovery_speed']
    out['fac_recovery_speed'] = [tbl.get(k, pop) for k in fac_k]
    out['ml_facility_effect_adj'] = [T['fac_effect'].get(k, 0.0) for k in fac_k]
    Z = T['scaler'].transform(out[cluster_cols])
    out['ml_cluster_id'] = T['kmeans'].predict(Z).astype(float)
    out['ml_cluster_dist'] = T['kmeans'].transform(Z).min(1)
    p = T['pca'].transform(Z)
    out['ml_pca1'], out['ml_pca2'] = p[:, 0], p[:, 1]
    out['ml_similar_children_avg'] = T['y_nn'][T['nn'].kneighbors(Z)[1]].mean(1)
    return out


def predict_members(members, X, fac_cols, use_facility=True):
    X = X.copy()
    if not use_facility and fac_cols:
        X[fac_cols] = np.nan
    pred = np.zeros(len(X))
    for v in (0., 1.):
        mk = X['sam_flag'].values == v
        if mk.any():
            pred[mk] = np.mean([m.predict(X.loc[mk, sub]) for sub, m in members[v]], axis=0)
    return pred


def predict_length_of_treatment(raw_input, bundle):
    """raw_input keys: sex, age_months, caregiver_age, whz, muac, haz, residence, rooms,
    maternal_edu, pneumonia, lost_appetite, ipc_phase, hunger_scale, handwashing (text
    labels or numbers), plus SAMMAM ('SAM'/'MAM'), RegioCEGn, facility (optional).
    Direct port of the notebook's own predict_one, since the notebook itself only ever
    saves the model/lookup objects, not a servable predict function.
    """
    m = bundle['maps']
    enc = {'sex': m['sex'], 'residence': m['residence'], 'pneumonia': m['yes_no'],
           'lost_appetite': m['yes_no'], 'ipc_phase': m['ipc'], 'hunger_scale': m['hunger']}

    row = {}
    for c in bundle['impute_cols']:
        v = raw_input.get(c)
        if isinstance(v, str):
            v = enc[c].get(v.strip().lower() if c in ('pneumonia', 'lost_appetite') else v)
        row[c] = np.nan if v is None else float(v)

    df = pd.DataFrame([row])
    filled = pd.DataFrame(bundle['imputer'].transform(df[bundle['impute_cols']]), columns=bundle['impute_cols'])
    for c in bundle['impute_cols']:
        filled[c] = filled[c].clip(*bundle['limits'][c])
        if c in bundle['discrete']:
            filled[c] = filled[c].round()
    filled = filled.where(df.isna(), df)

    X = add_derived(filled)
    sam_flag = 1.0 if raw_input['SAMMAM'] == 'SAM' else 0.0
    X['sam_flag'] = sam_flag

    facility = raw_input.get('facility')
    known = facility is not None and str(facility) in set(bundle['facilities'])
    fac_key = np.array([str(facility) if known else '__none__'])
    region_key = np.array([str(raw_input['RegioCEGn'])])
    sam_key = np.array([sam_flag])

    full = assemble(bundle['history_tables'], X, fac_key, region_key, sam_key, bundle['cluster_cols'])
    predicted_days = float(predict_members(
        bundle['members'], full[bundle['final_features'] + ['sam_flag']],
        bundle['facility_features'], use_facility=known)[0])

    return {'predicted_days': predicted_days, 'used_facility': known}
