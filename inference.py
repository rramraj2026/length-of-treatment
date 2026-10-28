import numpy as np
import pandas as pd


def build_feature_row(raw_input, bundle):
    """raw_input: dict of raw field values collected from the Streamlit form.
    Returns a one-row DataFrame with the exact design-matrix columns the model expects.
    """
    facility = raw_input['facility']
    sammam = raw_input['SAMMAM']
    region = raw_input['RegioCEGn']

    fac_type_key = f"{facility}_{sammam}" if facility else None
    facility_avg_duration_by_type_loo = bundle['facility_avg_duration_by_type_lookup'].get(
        fac_type_key, bundle['facility_avg_duration_by_type_pop'])
    facility_recovery_speed_loo = bundle['facility_recovery_speed_lookup'].get(
        facility, bundle['facility_recovery_speed_pop'])
    facility_relapse_rate_loo = bundle['facility_relapse_rate_lookup'].get(
        facility, bundle['facility_relapse_rate_pop'])
    region_avg_duration_loo = bundle['region_avg_duration_lookup'].get(
        region, bundle['region_avg_duration_pop'])

    # Household Vulnerability Index: IPC food security phase, household hunger scale,
    # and handwashing access, added together. Kept WITH the hunger scale for this model,
    # tested directly against the version without it, confirmed on 2026-10-27 to carry
    # a bit more real predictive weight here (the relapse model found the opposite).
    ipc_num = bundle['ipc_map'][raw_input['ipc_phase_fcs']]
    hhs_num = bundle['hhs_map'][raw_input['hhs_category']]
    vulnerability_index = ipc_num + hhs_num + (1 - raw_input['hndwash'])

    row = {
        'facility_avg_duration_by_type_loo': facility_avg_duration_by_type_loo,
        'facility_recovery_speed_loo': facility_recovery_speed_loo,
        'MUACofchildatadmission': raw_input['MUACofchildatadmission'],
        'region_avg_duration_loo': region_avg_duration_loo,
        'facility_relapse_rate_loo': facility_relapse_rate_loo,
        'whz_adm_clean': raw_input['whz_adm_clean'],
        'haz_adm_clean': raw_input['haz_adm_clean'],
        'household_rooms': raw_input['household_rooms'],
        'maternal_school_level': raw_input['maternal_school_level'],
        'vulnerability_index': vulnerability_index,
        'CaregiverssAge': raw_input['CaregiverssAge'],
        'admission_severity_tier_moderate': 1 if raw_input['admission_severity_tier'] == 'moderate' else 0,
        'admission_severity_tier_severe': 1 if raw_input['admission_severity_tier'] == 'severe' else 0,
        'residence_Urban-Host': 1 if raw_input['residence'] == 'Urban-Host' else 0,
        'residence_Urban-IDP': 1 if raw_input['residence'] == 'Urban-IDP' else 0,
        'cage_T0_cat_under 2 years': 1 if raw_input['cage_T0_cat'] == 'under 2 years' else 0,
        'hndwash_1.0': int(raw_input['hndwash']),
        'sex_male': 1 if raw_input['sex'] == 'male' else 0,
        'recent_pneumonia_yes': 1 if raw_input['recent_pneumonia'] == 'yes' else 0,
        'lost_appetite_yes': 1 if raw_input['lost_appetite'] == 'yes' else 0,
    }

    X = pd.DataFrame([row])
    X = X.reindex(columns=bundle['design_columns'], fill_value=0)
    return X


def predict_length_of_treatment(raw_input, bundle):
    X = build_feature_row(raw_input, bundle)
    model = bundle['model']
    predicted_days = float(model.predict(X)[0])

    return {
        'predicted_days': predicted_days,
    }
