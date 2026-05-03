import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model


st.set_page_config(
    page_title="Signal Desk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=JetBrains+Mono:wght@400;600&family=Manrope:wght@400;500;700&display=swap');

    :root {
        --bg: #111111;
        --paper: #f1e8d8;
        --ink: #121212;
        --muted: #5f574b;
        --line: rgba(18, 18, 18, 0.15);
        --red: #d9472b;
        --yellow: #e9bb39;
        --green: #17885c;
        --blue: #1c5fd4;
        --cream: #fffaf0;
        --shadow: 12px 12px 0 rgba(0, 0, 0, 0.22);
    }

    .stApp {
        background:
            linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(180deg, #141414 0%, #0d0d0d 100%);
        background-size: 32px 32px, 32px 32px, auto;
        font-family: 'Manrope', sans-serif;
        color: var(--paper);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
        letter-spacing: -0.04em;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
        color: #f0e7d8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .ticker {
        border: 1px solid rgba(241, 232, 216, 0.18);
        background: rgba(255, 255, 255, 0.02);
        padding: 0.55rem 0.8rem;
        min-height: 48px;
    }

    .hero-sheet,
    .info-sheet,
    .result-sheet,
    [data-testid="stForm"] {
        background: var(--paper);
        color: var(--ink);
        border: 2px solid #111;
        box-shadow: var(--shadow);
    }

    .hero-sheet {
        padding: 1.3rem;
        min-height: 100%;
    }

    .hero-kicker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--red);
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 4.4rem;
        line-height: 0.88;
        max-width: 9ch;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }

    .hero-copy {
        max-width: 34rem;
        font-size: 1rem;
        line-height: 1.6;
        color: var(--muted);
        margin-bottom: 1.4rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
    }

    .hero-stat {
        border: 1.5px solid #111;
        padding: 0.85rem;
        background: rgba(255, 255, 255, 0.42);
    }

    .hero-stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }

    .hero-stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem;
        text-transform: uppercase;
    }

    .info-sheet {
        padding: 1.3rem;
        height: 100%;
    }

    .info-tag {
        display: inline-block;
        padding: 0.28rem 0.5rem;
        background: var(--yellow);
        border: 1.5px solid #111;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .info-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        line-height: 1;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }

    .info-copy {
        color: var(--muted);
        line-height: 1.65;
        margin-bottom: 1rem;
    }

    .info-list {
        display: grid;
        gap: 0.6rem;
    }

    .info-item {
        border-left: 4px solid #111;
        padding-left: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    [data-testid="stForm"] {
        margin-top: 1.5rem;
        padding: 1rem 1rem 0.6rem 1rem;
    }

    [data-testid="stForm"] h3 {
        font-size: 1.35rem;
        text-transform: uppercase;
    }

    label, .stNumberInput label, .stSelectbox label {
        font-weight: 700 !important;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: var(--cream);
        border: 1.5px solid #111 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"],
    [data-testid="stSelectbox"] {
        margin-bottom: 0.55rem;
    }

    .stFormSubmitButton > button {
        width: 100%;
        min-height: 56px;
        background: var(--red);
        color: #fff7ef;
        border: 2px solid #111;
        border-radius: 0;
        box-shadow: 8px 8px 0 rgba(0, 0, 0, 0.2);
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stFormSubmitButton > button:hover {
        background: #bc3118;
        color: white;
        border: 2px solid #111;
    }

    .result-sheet {
        margin-top: 1.4rem;
        padding: 1.1rem;
    }

    .result-band {
        display: inline-block;
        margin-bottom: 0.9rem;
        padding: 0.35rem 0.55rem;
        border: 2px solid #111;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
    }

    .band-yes {
        background: #b9f0d7;
    }

    .band-no {
        background: #ffd0c6;
    }

    .result-headline {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        line-height: 0.95;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        max-width: 14ch;
    }

    .result-text {
        color: var(--muted);
        max-width: 42rem;
        line-height: 1.7;
    }

    .divider-note {
        margin-top: 0.9rem;
        padding-top: 0.9rem;
        border-top: 1px dashed rgba(18,18,18,0.22);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        color: var(--muted);
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 2.8rem;
            max-width: none;
        }

        .hero-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model():
    return load_model("loan_model")


model = load_trained_model()


def build_input_frame(values):
    return pd.DataFrame(
        {
            "age": [values["age"]],
            "job": [values["job"]],
            "marital": [values["marital"]],
            "education": [values["education"]],
            "default": [values["default"]],
            "housing": [values["housing"]],
            "loan": [values["loan"]],
            "contact": [values["contact"]],
            "month": [values["month"]],
            "day_of_week": [values["day_of_week"]],
            "duration": [values["duration"]],
            "campaign": [values["campaign"]],
            "pdays": [values["pdays"]],
            "previous": [values["previous"]],
            "poutcome": [values["poutcome"]],
            "emp.var.rate": [values["emp_var_rate"]],
            "cons.price.idx": [values["cons_price_idx"]],
            "cons.conf.idx": [values["cons_conf_idx"]],
            "euribor3m": [values["euribor3m"]],
            "nr.employed": [values["nr_employed"]],
        }
    )


st.markdown(
    """
    <div class="topbar">
        <div class="ticker">Signal Desk / Retail Deposit Classifier / Internal Tool</div>
        <div class="ticker">One-page underwriting style decision surface</div>
    </div>
    """,
    unsafe_allow_html=True,
)


hero_col, info_col = st.columns([1.45, 0.75], gap="large")

with hero_col:
    st.markdown(
        """
        <div class="hero-sheet">
            <div class="hero-kicker">Campaign Conversion Control Room</div>
            <div class="hero-title">Read the client. Call the signal.</div>
            <div class="hero-copy">
                This screen is built to feel more like a trading desk than a generated app.
                Put in the client and campaign details, then get one decisive outcome on deposit intent.
            </div>
            <div class="hero-grid">
                <div class="hero-stat">
                    <div class="hero-stat-label">Focus</div>
                    <div class="hero-stat-value">Term Deposit</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Decision</div>
                    <div class="hero-stat-value">Yes or No</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Mode</div>
                    <div class="hero-stat-value">Fast Signal</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info_col:
    st.markdown(
        """
        <div class="info-sheet">
            <div class="info-tag">Operator Note</div>
            <div class="info-title">Not another soft landing page.</div>
            <div class="info-copy">
                The design intentionally avoids the usual rounded-glass AI look.
                It uses hard edges, paper contrast, and mono labels so the interface has more character.
            </div>
            <div class="info-list">
                <div class="info-item">Profile inputs on top</div>
                <div class="info-item">Campaign signals below</div>
                <div class="info-item">Single high-clarity output</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.form("prediction_form"):
    st.subheader("Client Profile")
    left_a, left_b, left_c = st.columns(3)

    with left_a:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        default = st.selectbox("Credit Default", ["yes", "no", "unknown"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])

    with left_b:
        job = st.selectbox(
            "Job",
            [
                "admin.",
                "technician",
                "services",
                "management",
                "retired",
                "blue-collar",
                "unemployed",
                "entrepreneur",
                "housemaid",
                "student",
                "self-employed",
                "unknown",
            ],
        )
        education = st.selectbox(
            "Education",
            [
                "basic.4y",
                "basic.6y",
                "basic.9y",
                "high.school",
                "professional.course",
                "university.degree",
                "illiterate",
                "unknown",
            ],
        )
        housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
        month = st.selectbox(
            "Month",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        )

    with left_c:
        loan = st.selectbox("Personal Loan", ["yes", "no", "unknown"])
        day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
        poutcome = st.selectbox("Previous Outcome", ["failure", "nonexistent", "success"])
        previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0)

    st.subheader("Campaign and Market Signals")
    right_a, right_b, right_c = st.columns(3)

    with right_a:
        duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=100)
        campaign = st.number_input("Number of Contacts", min_value=1, max_value=50, value=1)
        pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=999, value=-1)

    with right_b:
        emp_var_rate = st.number_input("Employment Variation Rate", min_value=-5.0, max_value=5.0, value=1.1)
        cons_price_idx = st.number_input("Consumer Price Index", min_value=90.0, max_value=100.0, value=93.0)
        cons_conf_idx = st.number_input("Consumer Confidence Index", min_value=-60.0, max_value=0.0, value=-40.0)

    with right_c:
        euribor3m = st.number_input("Euribor 3 Month Rate", min_value=0.0, max_value=10.0, value=4.0)
        nr_employed = st.number_input("Number of Employees", min_value=4000.0, max_value=6000.0, value=5000.0)
        submitted = st.form_submit_button("Run Signal Check")


if submitted:
    input_data = build_input_frame(
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp_var_rate": emp_var_rate,
            "cons_price_idx": cons_price_idx,
            "cons_conf_idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr_employed": nr_employed,
        }
    )

    try:
        prediction = predict_model(model, data=input_data)
        result = prediction["prediction_label"][0]
        is_positive = result == 1 or str(result).lower() == "yes"

        band_class = "band-yes" if is_positive else "band-no"
        band_text = "Subscription Signal: Positive" if is_positive else "Subscription Signal: Negative"
        headline = "This client looks ready for the deposit pitch." if is_positive else "This client does not look ready for the deposit pitch."
        copy = (
            "The entered profile matches a stronger conversion pattern. Treat this as a prioritization signal for outreach."
            if is_positive
            else "The entered profile matches a weaker conversion pattern. Reframe the offer or deprioritize this lead."
        )

        st.markdown(
            f"""
            <div class="result-sheet">
                <div class="result-band {band_class}">{band_text}</div>
                <div class="result-headline">{headline}</div>
                <div class="result-text">{copy}</div>
                <div class="divider-note">Model output simplified for quick decision use</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
