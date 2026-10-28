import os

import joblib
import streamlit as st

from inference import predict_length_of_treatment

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model_bundle.joblib")

CARD_BG = "#111827"
RESULT_ACCENT = "#3B82F6"
SECTION_ACCENTS = {"admission": "#3B82F6", "health": "#06B6D4", "food": "#22C55E"}


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def section_card_open(accent):
    st.markdown(
        f"""
        <div class="section-card" style="border-left: 3px solid {accent}; background: rgba(255,255,255,0.03);
            border-radius: 0 14px 14px 0; padding: 16px 18px 10px 18px; margin-bottom: 10px;
            transition: all 0.3s ease;">
        """,
        unsafe_allow_html=True,
    )


def section_card_close(filled, total, accent):
    done = filled == total
    badge_color = "#22C55E" if done else "#94A3B8"
    icon = "✅" if done else "🟡"
    bar_color = "#22C55E" if done else accent
    pct = int(100 * filled / total)
    st.markdown(
        f"""
            <div style="margin-top:6px;">
                <div style="background:rgba(255,255,255,0.08); border-radius:999px; height:6px; overflow:hidden;">
                    <div style="width:{pct}%; background:{bar_color}; height:100%; border-radius:999px;
                        transition: width 0.4s ease;"></div>
                </div>
                <p style="text-align:right; font-size:12px; color:{badge_color}; margin:4px 0 0 0;">
                    {icon} {filled}/{total} filled
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def whz_tag(value):
    if value is None:
        return
    if value < -3:
        st.caption("🔴 Severe wasting range (WHZ < -3)")
    elif value < -2:
        st.caption("🟠 Moderate wasting range (-3 to -2)")
    else:
        st.caption("🟢 Within normal range (≥ -2)")


def muac_tag(value):
    if value is None:
        return
    if value < 11.5:
        st.caption("🔴 Severe wasting range (< 11.5 cm)")
    elif value < 12.5:
        st.caption("🟠 Moderate wasting range (11.5–12.5 cm)")
    else:
        st.caption("🟢 Within normal range (≥ 12.5 cm)")


def haz_tag(value):
    if value is None:
        return
    if value < -3:
        st.caption("🔴 Severe stunting range (HAZ < -3)")
    elif value < -2:
        st.caption("🟠 Moderate stunting range (-3 to -2)")
    else:
        st.caption("🟢 Within normal range (≥ -2)")


st.set_page_config(page_title="Length of Treatment Predictor", layout="centered", page_icon="⏳")


def inject_global_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 15% 20%, rgba(59,130,246,0.18) 0%, rgba(59,130,246,0) 40%),
                radial-gradient(circle at 85% 15%, rgba(139,92,246,0.16) 0%, rgba(139,92,246,0) 40%),
                radial-gradient(circle at 25% 85%, rgba(6,182,212,0.14) 0%, rgba(6,182,212,0) 40%),
                radial-gradient(circle at 80% 80%, rgba(236,72,153,0.12) 0%, rgba(236,72,153,0) 40%),
                #0B1220;
            background-size: 180% 180%, 180% 180%, 180% 180%, 180% 180%, 100% 100%;
            animation: driftBg 22s ease-in-out infinite alternate;
        }
        @keyframes driftBg {
            0%   { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 0 0; }
            50%  { background-position: 30% 20%, 70% 30%, 20% 70%, 80% 90%, 0 0; }
            100% { background-position: 10% 40%, 90% 10%, 40% 90%, 60% 70%, 0 0; }
        }
        .section-card:hover {
            background: rgba(255,255,255,0.055) !important;
            transform: translateX(2px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_style()


@st.cache_resource
def load_bundle():
    return joblib.load(ARTIFACT_PATH)


bundle = load_bundle()
cat = bundle["category_levels"]
rng = bundle["raw_ranges"]

st.markdown(
    """
    <div style="padding: 18px 0 6px 0;">
        <h1 style="margin-bottom:4px; color:#F1F5F9;">⏳ Length of Treatment Predictor</h1>
        <p style="color:#94A3B8; font-size:15px; margin-top:0;">
            Estimates how many days a child's treatment is likely to take, using only information
            known at admission.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.header("At admission")
section_card_open(SECTION_ACCENTS["admission"])

c1, c2 = st.columns(2)
with c1:
    facility = st.selectbox("Facility (Optional)", bundle["facility_list"], index=None, placeholder="Select...")
    sammam = st.selectbox("Admission type (SAM / MAM)", ["SAM", "MAM"], index=None, placeholder="Select...")
    region = st.selectbox(
        "Region", bundle["region_list"], index=None, placeholder="Select...",
        format_func=lambda v: v.replace("_", " "))
    sex = st.selectbox(
        "Child's sex", cat["sex"], index=None, placeholder="Select...",
        format_func=lambda v: v.capitalize())
    cage_cat = st.selectbox(
        "Age category at admission", cat["cage_T0_cat"], index=None, placeholder="Select...")
    caregiver_age = st.number_input(
        "Caregiver's age (years)",
        min_value=float(rng["CaregiverssAge"][0]), max_value=float(rng["CaregiverssAge"][1]),
        value=None, placeholder=f"e.g. {rng['CaregiverssAge'][0]:.0f}-{rng['CaregiverssAge'][1]:.0f}")

with c2:
    whz_adm = st.number_input(
        "Weight-for-Height Z-score at admission",
        min_value=float(rng["whz_adm_clean"][0]), max_value=float(rng["whz_adm_clean"][1]),
        value=None, placeholder=f"e.g. {rng['whz_adm_clean'][0]:.1f} to {rng['whz_adm_clean'][1]:.1f}")
    whz_tag(whz_adm)
    muac_adm = st.number_input(
        "Mid-Upper Arm Circumference at admission (cm)",
        min_value=float(rng["MUACofchildatadmission"][0]), max_value=float(rng["MUACofchildatadmission"][1]),
        value=None, placeholder=f"e.g. {rng['MUACofchildatadmission'][0]:.1f}-{rng['MUACofchildatadmission'][1]:.1f} cm")
    muac_tag(muac_adm)
    haz_adm = st.number_input(
        "Height-for-Age Z-score at admission",
        min_value=float(rng["haz_adm_clean"][0]), max_value=float(rng["haz_adm_clean"][1]),
        value=None, placeholder=f"e.g. {rng['haz_adm_clean'][0]:.1f} to {rng['haz_adm_clean'][1]:.1f}")
    haz_tag(haz_adm)
    residence = st.selectbox("Residence type", cat["residence"], index=None, placeholder="Select...")
    severity_tier = st.selectbox(
        "Admission severity tier", cat["admission_severity_tier"], index=None, placeholder="Select...",
        format_func=lambda v: v.capitalize())

admission_vals = [facility, sammam, region, sex, cage_cat, caregiver_age, whz_adm, muac_adm, haz_adm,
                   residence, severity_tier]
section_card_close(sum(v is not None for v in admission_vals), len(admission_vals), SECTION_ACCENTS["admission"])

st.subheader("Household and recent health")
section_card_open(SECTION_ACCENTS["health"])
c3, c4 = st.columns(2)
with c3:
    household_rooms = st.number_input(
        "Number of rooms in household",
        min_value=float(rng["household_rooms"][0]), max_value=float(rng["household_rooms"][1]),
        value=None, placeholder=f"e.g. {rng['household_rooms'][0]:.0f}-{rng['household_rooms'][1]:.0f}")
    maternal_school_level = st.number_input(
        "Maternal education level (0 = no schooling, 4 = highest level recorded in this survey)",
        min_value=float(rng["maternal_school_level"][0]), max_value=float(rng["maternal_school_level"][1]),
        value=None, step=1.0, placeholder="0-4")
with c4:
    pneumonia = st.selectbox(
        "Pneumonia in the last 2 weeks", cat["recent_pneumonia"], index=None, placeholder="Select...",
        format_func=lambda v: v.capitalize())
    appetite = st.selectbox(
        "Lost appetite in the last 2 weeks", cat["lost_appetite"], index=None, placeholder="Select...",
        format_func=lambda v: v.capitalize())

health_vals = [household_rooms, maternal_school_level, pneumonia, appetite]
section_card_close(sum(v is not None for v in health_vals), len(health_vals), SECTION_ACCENTS["health"])

st.subheader("Household food security")
section_card_open(SECTION_ACCENTS["food"])
ipc_phase = st.selectbox(
    "IPC food security phase", list(bundle["ipc_map"].keys()), index=None, placeholder="Select...")
hhs_category = st.selectbox(
    "Household hunger scale", list(bundle["hhs_map"].keys()), index=None, placeholder="Select...")
hndwash_label = st.selectbox("Household handwashing access", ["Yes", "No"], index=None, placeholder="Select...")

food_vals = [ipc_phase, hhs_category, hndwash_label]
section_card_close(sum(v is not None for v in food_vals), len(food_vals), SECTION_ACCENTS["food"])

st.divider()

if st.button("Predict length of treatment", type="primary"):
    raw_input = {
        "facility": facility,
        "SAMMAM": sammam,
        "RegioCEGn": region,
        "sex": sex,
        "cage_T0_cat": cage_cat,
        "CaregiverssAge": caregiver_age,
        "whz_adm_clean": whz_adm,
        "MUACofchildatadmission": muac_adm,
        "haz_adm_clean": haz_adm,
        "residence": residence,
        "admission_severity_tier": severity_tier,
        "household_rooms": household_rooms,
        "maternal_school_level": maternal_school_level,
        "recent_pneumonia": pneumonia,
        "lost_appetite": appetite,
        "ipc_phase_fcs": ipc_phase,
        "hhs_category": hhs_category,
        "hndwash": None if hndwash_label is None else (1 if hndwash_label == "Yes" else 0),
    }

    field_labels = {
        "facility": "Facility", "SAMMAM": "Admission type", "RegioCEGn": "Region",
        "sex": "Child's sex", "cage_T0_cat": "Age category at admission",
        "CaregiverssAge": "Caregiver's age", "whz_adm_clean": "WHZ at admission",
        "MUACofchildatadmission": "MUAC at admission", "haz_adm_clean": "HAZ at admission",
        "residence": "Residence type", "admission_severity_tier": "Admission severity tier",
        "household_rooms": "Number of rooms in household",
        "maternal_school_level": "Maternal education level",
        "recent_pneumonia": "Pneumonia in the last 2 weeks", "lost_appetite": "Lost appetite in the last 2 weeks",
        "ipc_phase_fcs": "IPC food security phase", "hhs_category": "Household hunger scale",
        "hndwash": "Household handwashing access",
    }
    # Facility is optional, left blank, the model falls back to the population-average
    # facility numbers instead of blocking the prediction.
    missing = [field_labels[k] for k, v in raw_input.items() if v is None and k != "facility"]

    range_check_fields = {
        "CaregiverssAge": "CaregiverssAge", "whz_adm_clean": "whz_adm_clean",
        "MUACofchildatadmission": "MUACofchildatadmission", "haz_adm_clean": "haz_adm_clean",
        "household_rooms": "household_rooms", "maternal_school_level": "maternal_school_level",
    }
    out_of_range = []
    if not missing:
        for field_key, range_key in range_check_fields.items():
            value = raw_input[field_key]
            lo, hi = rng[range_key]
            if value < lo or value > hi:
                out_of_range.append(
                    f"{field_labels[field_key]} = {value} (valid range: {lo:.1f} to {hi:.1f})"
                )

    if missing:
        st.error(
            "Please fill in every field before predicting, the model isn't allowed to guess. "
            "Missing: " + ", ".join(missing)
        )
    elif out_of_range:
        st.error(
            "This model was never trained on values outside the ranges below, so it can't "
            "give a reliable estimate here. Out of range: " + "; ".join(out_of_range)
        )
    else:
        result = predict_length_of_treatment(raw_input, bundle)
        color = RESULT_ACCENT
        glow_dim = hex_to_rgba(color, 0.22)
        glow_bright = hex_to_rgba(color, 0.5)

        st.markdown(
            f"""
            <style>
            @keyframes pulseGlowLOT {{
                0%   {{ box-shadow: 0 0 25px {glow_dim}; }}
                50%  {{ box-shadow: 0 0 60px {glow_bright}; }}
                100% {{ box-shadow: 0 0 25px {glow_dim}; }}
            }}
            </style>
            <div style="background:{CARD_BG}; border:1px solid rgba(255,255,255,0.08);
                border-radius:18px; padding:34px 10px; margin-top:8px; text-align:center;
                animation: pulseGlowLOT 2.4s ease-in-out infinite;">
                <div style="font-size:15px; color:#94A3B8;">Expected length of treatment</div>
                <div style="font-size:52px; font-weight:800; color:{color}; margin-top:4px;">
                    {result['predicted_days']:.0f} days
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This is an estimate, actual treatment length depends on how the child responds."
        )
