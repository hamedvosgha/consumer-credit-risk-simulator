
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="End-to-End Non-Bank Credit Risk Simulator",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
def logistic(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def cumulative_pd(h):
    h = np.clip(np.asarray(h), 0, 0.95)
    return 1 - np.cumprod(1 - h)

def psi(expected, actual, bins=10):
    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, q))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    a = pd.cut(actual, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    e = np.clip(e.values, 1e-6, None)
    a = np.clip(a.values, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

@st.cache_data
def make_synthetic_portfolio(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "age": rng.integers(21, 70, n),
        "income": np.exp(rng.normal(np.log(85000), 0.45, n)),
        "loan_amount": np.exp(rng.normal(np.log(22000), 0.55, n)),
        "term_months": rng.choice([24, 36, 48, 60], n, p=[0.12, 0.35, 0.33, 0.20]),
        "bureau_score": np.clip(rng.normal(690, 75, n), 350, 850),
        "dti": np.clip(rng.beta(2.1, 4.0, n) * 0.9, 0.02, 0.85),
        "prior_delinquency": rng.binomial(1, 0.14, n),
        "employment_years": np.clip(rng.normal(6.5, 5.0, n), 0, 35),
        "product": rng.choice(["Personal Loan", "Auto Loan", "Credit Line"], n, p=[0.48, 0.34, 0.18]),
        "vintage": rng.choice(
            ["2025 Q1","2025 Q2","2025 Q3","2025 Q4","2026 Q1","2026 Q2","2026 Q3"],
            n, p=[0.10,0.11,0.12,0.13,0.15,0.18,0.21]
        )
    })

    z = (
        -3.15
        - 0.0105 * (df["bureau_score"] - 650)
        + 2.45 * (df["dti"] - 0.30)
        + 0.78 * df["prior_delinquency"]
        + 0.000010 * (df["loan_amount"] - 20000)
        - 0.018 * (df["employment_years"] - 5)
    )
    df["orig_pd"] = logistic(z)
    df["score"] = np.clip(850 - 280 * df["orig_pd"] + rng.normal(0, 22, n), 300, 850)
    df["ead"] = df["loan_amount"] * rng.uniform(0.72, 1.0, n)
    df["lgd"] = np.clip(rng.normal(0.45, 0.10, n), 0.15, 0.80)
    df["dpd"] = rng.choice([0,15,30,60,90], n, p=[0.78,0.08,0.07,0.045,0.025])
    return df

# -----------------------------
# Global controls
# -----------------------------
st.title("End-to-End Consumer Credit Risk Simulator")
st.caption(
    "Synthetic teaching / research prototype covering origination, decisioning, pricing, "
    "behavioural risk, vintage analysis, forward-looking PD, AASB 9 / IFRS 9 ECL, "
    "portfolio loss distribution and model monitoring."
)

with st.sidebar:
    st.header("Global Scenario Controls")

    st.subheader("Macroeconomy")
    unemployment = st.slider("Unemployment rate (%)", 3.0, 10.0, 4.5, 0.1)
    gdp_growth = st.slider("GDP growth (%)", -4.0, 5.0, 2.0, 0.1)
    cash_rate = st.slider("Cash / interest rate (%)", 1.0, 9.0, 4.0, 0.1)
    inflation = st.slider("Inflation (%)", 1.0, 8.0, 2.5, 0.1)
    house_price = st.slider("House price growth (%)", -15.0, 15.0, 2.0, 0.5)

    st.subheader("Applicant population")
    applicant_volume = st.slider("Applicant volume", 1000, 20000, 6000, 500)
    nonprime_share = st.slider("Non-prime share (%)", 10, 80, 35, 1)
    drift_shift = st.slider("Population drift / score shift", -80, 80, 0, 5)

    st.subheader("Credit strategy")
    approve_cutoff = st.slider("Auto-approve score cut-off", 450, 780, 650, 5)
    review_band = st.slider("Manual review band (points)", 0, 80, 30, 5)
    max_pd = st.slider("Maximum acceptable PD (%)", 1.0, 25.0, 10.0, 0.5) / 100
    base_rate = st.slider("Base customer rate (%)", 4.0, 20.0, 9.0, 0.25)

    st.subheader("AASB 9 / IFRS 9")
    sicr_multiplier = st.slider("SICR: current/origination PD ratio", 1.2, 4.0, 2.0, 0.1)
    lgd_assumption = st.slider("Portfolio LGD (%)", 20, 80, 45, 1) / 100
    remaining_term = st.slider("Remaining term (months)", 12, 84, 48, 6)

    st.subheader("Scenario weights")
    w_up = st.slider("Upside weight", 0.0, 0.5, 0.20, 0.05)
    w_down = st.slider("Downside weight", 0.0, 0.5, 0.20, 0.05)
    w_base = max(0.0, 1.0 - w_up - w_down)
    st.write(f"Base weight: **{w_base:.0%}**")

    st.subheader("Portfolio loss simulation")
    asset_corr = st.slider("Default correlation", 0.01, 0.30, 0.08, 0.01)
    n_sims = st.select_slider("Monte Carlo simulations", options=[2000, 5000, 10000, 20000], value=5000)

# -----------------------------
# Data and scenario transforms
# -----------------------------
df = make_synthetic_portfolio(max(6000, applicant_volume))
df = df.iloc[:applicant_volume].copy()

# Alter risk mix synthetically
rng = np.random.default_rng(123)
nonprime_target = nonprime_share / 100
rank = df["score"].rank(pct=True)
nonprime_flag = rank <= nonprime_target
df.loc[nonprime_flag, "bureau_score"] -= 35
df["score_current"] = np.clip(df["score"] + drift_shift + rng.normal(0, 8, len(df)), 300, 850)

macro_index = (
    0.18 * (unemployment - 4.5)
    - 0.11 * (gdp_growth - 2.0)
    + 0.055 * (cash_rate - 4.0)
    + 0.025 * (inflation - 2.5)
    - 0.022 * (house_price - 2.0)
)
risk_mix_index = 0.95 * (nonprime_target - 0.35)

df["current_pd"] = logistic(logit(df["orig_pd"]) + 1.10 * macro_index + 0.80 * risk_mix_index)

approve_floor = approve_cutoff
review_floor = approve_cutoff - review_band
df["decision"] = np.where(
    (df["score_current"] >= approve_floor) & (df["current_pd"] <= max_pd),
    "Approve",
    np.where(
        (df["score_current"] >= review_floor) & (df["current_pd"] <= max_pd * 1.25),
        "Refer",
        "Decline"
    )
)

df["risk_margin"] = 0.015 + 2.7 * df["current_pd"] + 0.55 * df["lgd"] * df["current_pd"]
df["customer_rate"] = base_rate / 100 + df["risk_margin"]
df["expected_loss"] = df["ead"] * df["current_pd"] * lgd_assumption
df["expected_interest"] = df["ead"] * df["customer_rate"]
df["expected_contribution"] = df["expected_interest"] - df["expected_loss"] - 0.018 * df["ead"]

pd_ratio = df["current_pd"] / np.clip(df["orig_pd"], 1e-6, None)
df["stage"] = np.select(
    [df["dpd"] >= 90, (pd_ratio >= sicr_multiplier) | (df["dpd"] >= 30)],
    [3, 2],
    default=1
)

approval_rate = (df["decision"] == "Approve").mean()
referral_rate = (df["decision"] == "Refer").mean()
decline_rate = (df["decision"] == "Decline").mean()
approved_bad_rate = df.loc[df["decision"]=="Approve", "current_pd"].mean()
psi_value = psi(df["score"], df["score_current"])

# -----------------------------
# Executive KPIs
# -----------------------------
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Approval rate", f"{approval_rate:.1%}")
k2.metric("Referral rate", f"{referral_rate:.1%}")
k3.metric("Approved avg PD", f"{approved_bad_rate:.2%}")
k4.metric("Stage 2+", f"{(df['stage']>=2).mean():.1%}")
k5.metric("Portfolio EAD", f"${df['ead'].sum()/1e6:,.1f}m")
k6.metric("PSI", f"{psi_value:.3f}")

if psi_value >= 0.25:
    st.error("Model monitoring alert: substantial population shift. Model review / recalibration is recommended.")
elif psi_value >= 0.10:
    st.warning("Model monitoring alert: moderate population shift. Investigate drivers and monitor closely.")
else:
    st.success("Population stability is currently within the illustrative monitoring range.")

tabs = st.tabs([
    "README / About",
    "Executive Dashboard",
    "Origination & Decisioning",
    "Pricing & Profitability",
    "Vintage & Roll Rates",
    "Behavioural / Hazard",
    "Forward-Looking PD",
    "AASB 9 / ECL",
    "Loss Distribution",
    "Model Monitoring"
])

# =========================================================
# 0. README / ABOUT
# =========================================================
with tabs[0]:
    st.subheader("About this simulator")
    st.markdown(
        """
        **Developed by Hamed Vosgha and Jean-Pierre Fenech**

        This interactive simulator has been developed as a **proposal and showcase**
        of an end-to-end consumer credit risk analytics framework.

        It demonstrates how credit-risk modelling, lending strategy, pricing,
        behavioural monitoring, macroeconomic scenarios, AASB 9 / IFRS 9 staging
        and ECL, portfolio loss simulation, vintage analysis and model monitoring
        can be brought together in one interactive decision-support environment.

        All data, model relationships and outputs in this prototype are **synthetic
        and illustrative**. The application is intended for research, teaching and
        project-development discussion rather than production lending decisions or
        financial reporting.
        """
    )

# =========================================================
# 1. EXECUTIVE DASHBOARD
# =========================================================
with tabs[2]:
    st.subheader("Credit-risk lifecycle")
    st.markdown(
        "**Applicants → Policy & Affordability → Score / PD → Decision & Pricing → "
        "Origination → Behavioural Monitoring → SICR / Staging → Lifetime PD/LGD/EAD → "
        "ECL → Portfolio Risk & Profitability**"
    )

    c1, c2 = st.columns(2)
    with c1:
        dec = df["decision"].value_counts().reindex(["Approve","Refer","Decline"]).fillna(0).reset_index()
        dec.columns = ["Decision","Count"]
        fig = px.bar(dec, x="Decision", y="Count", title="Decision outcomes")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        stage_counts = df["stage"].value_counts().sort_index().reset_index()
        stage_counts.columns = ["Stage","Count"]
        fig = px.pie(stage_counts, names="Stage", values="Count", title="AASB 9 stage mix")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What changed?")
    baseline_macro = 0.0
    delta_pd = df["current_pd"].mean() - df["orig_pd"].mean()
    st.info(
        f"Current scenario moves average PD by **{delta_pd:+.2%}** relative to origination, "
        f"with **{(df['stage']>=2).mean():.1%}** of exposures now in Stage 2 or 3. "
        f"The current applicant population PSI is **{psi_value:.3f}**."
    )

# =========================================================
# 2. ORIGINATION & DECISIONING
# =========================================================
with tabs[2]:
    st.subheader("Origination score distribution and policy cut-offs")

    fig = px.histogram(
        df, x="score_current", color="decision", nbins=45,
        title="Current applicant score distribution by decision"
    )
    fig.add_vline(x=approve_cutoff, line_dash="dash", annotation_text="Approve cut-off")
    fig.add_vline(x=review_floor, line_dash="dot", annotation_text="Review floor")
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Approve", f"{approval_rate:.1%}")
    c2.metric("Refer", f"{referral_rate:.1%}")
    c3.metric("Decline", f"{decline_rate:.1%}")

    st.subheader("Risk around the decision boundary")
    band = df[(df["score_current"] >= review_floor - 20) & (df["score_current"] <= approve_cutoff + 20)].copy()
    band["score_band"] = pd.cut(band["score_current"], bins=8)
    local = band.groupby("score_band", observed=False).agg(
        applicants=("loan_id","count"),
        avg_pd=("current_pd","mean"),
        approval_rate=("decision", lambda x: (x=="Approve").mean())
    ).reset_index()
    local["score_band"] = local["score_band"].astype(str)
    st.dataframe(local, use_container_width=True)

    st.caption(
        "This section is deliberately focused on the region near the decision boundary, "
        "because a model can rank the full population well while still making weak decisions close to the cut-off."
    )

# =========================================================
# 3. PRICING & PROFITABILITY
# =========================================================
with tabs[3]:
    st.subheader("Risk-based pricing and expected contribution")

    approved = df[df["decision"]=="Approve"].copy()
    if len(approved):
        fig = px.scatter(
            approved.sample(min(1800, len(approved)), random_state=1),
            x="current_pd", y="customer_rate", size="ead", color="product",
            labels={"current_pd":"Current PD","customer_rate":"Customer rate"},
            title="Pricing response to credit risk"
        )
        st.plotly_chart(fig, use_container_width=True)

        summary = approved.groupby("product").agg(
            loans=("loan_id","count"),
            avg_pd=("current_pd","mean"),
            avg_rate=("customer_rate","mean"),
            expected_loss=("expected_loss","sum"),
            expected_contribution=("expected_contribution","sum")
        ).reset_index()
        st.dataframe(summary, use_container_width=True)
    else:
        st.warning("No approved accounts under the current policy settings.")

# =========================================================
# 4. VINTAGE & ROLL RATES
# =========================================================
with tabs[4]:
    st.subheader("Vintage / cohort delinquency development")

    vintages = ["2025 Q1","2025 Q2","2025 Q3","2025 Q4","2026 Q1","2026 Q2","2026 Q3"]
    mobs = np.arange(1, 13)
    heat = []
    for i, v in enumerate(vintages):
        vintage_risk = 0.02 + 0.006*i + 0.025*max(0, macro_index)
        for m in mobs:
            rate_30 = np.clip(vintage_risk + 0.0045*m + 0.002*np.sin(m/2+i), 0.005, 0.45)
            heat.append([v,m,rate_30])
    heat_df = pd.DataFrame(heat, columns=["Vintage","MOB","30+ DPD Rate"])

    pivot = heat_df.pivot(index="Vintage", columns="MOB", values="30+ DPD Rate")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorbar=dict(title="30+ DPD")
        )
    )
    fig.update_layout(
        title="30+ DPD vintage heatmap",
        xaxis_title="Months on Book",
        yaxis_title="Vintage"
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_vintage = st.selectbox("Select vintage for roll-rate matrix", vintages, index=len(vintages)-1)

    deterioration = max(0, macro_index) + 0.3 * max(0, nonprime_target - 0.35)
    roll = pd.DataFrame(
        [
            [0.90-0.04*deterioration, 0.06+0.015*deterioration, 0.025, 0.010, 0.005],
            [0.35, 0.42-0.03*deterioration, 0.16+0.02*deterioration, 0.05, 0.02],
            [0.16, 0.10, 0.48-0.04*deterioration, 0.20+0.02*deterioration, 0.06+0.02*deterioration],
            [0.06, 0.04, 0.08, 0.52-0.03*deterioration, 0.30+0.03*deterioration],
            [0.015, 0.005, 0.01, 0.02, 0.95]
        ],
        index=["Current","1–29 DPD","30–59 DPD","60–89 DPD","90+ / Default"],
        columns=["Current","1–29 DPD","30–59 DPD","60–89 DPD","90+ / Default"]
    )
    roll = roll.div(roll.sum(axis=1), axis=0)
    st.dataframe((roll*100).round(1).astype(str) + "%", use_container_width=True)

    st.caption(
        "Illustrative roll rates show movement from Current → 1–29 → 30–59 → 60–89 → 90+/Default, "
        "including cures back to better delinquency states."
    )

# =========================================================
# 5. BEHAVIOURAL / HAZARD
# =========================================================
with tabs[5]:
    st.subheader("Behavioural risk, hazard and survival")

    months = np.arange(1, remaining_term+1)
    base_annual_pd = np.clip(df["current_pd"].mean(), 0.003, 0.35)
    seasoning = 0.15*np.sin((months-4)/8) + 0.008*months
    annual_curve = logistic(logit(base_annual_pd) + seasoning)
    monthly_hazard = 1 - (1-annual_curve)**(1/12)
    survival = np.cumprod(1-monthly_hazard)

    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=monthly_hazard, name="Monthly hazard"))
        fig.update_layout(title="Monthly default hazard", xaxis_title="Month", yaxis_title="Hazard")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=survival, name="Survival"))
        fig.update_layout(title="Survival curve", xaxis_title="Month", yaxis_title="Survival probability")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 6. FORWARD-LOOKING PD
# =========================================================
with tabs[6]:
    st.subheader("Forward-looking scenario PD term structures")

    months = np.arange(1, remaining_term+1)
    pd0 = np.clip(df["current_pd"].mean(), 0.003, 0.30)

    scenario_shift = {
        "Upside": -0.45,
        "Base": 0.00,
        "Downside": 0.70 + max(0, macro_index)
    }

    scenario_hazards = {}
    cumulative = {}
    for name, shift in scenario_shift.items():
        annual_pd_t = logistic(logit(pd0) + shift + 0.006*months)
        h = 1 - (1-annual_pd_t)**(1/12)
        scenario_hazards[name] = h
        cumulative[name] = cumulative_pd(h)

    fig = go.Figure()
    for name in ["Upside","Base","Downside"]:
        fig.add_trace(go.Scatter(x=months, y=cumulative[name], name=name))
    fig.update_layout(
        title="Cumulative PD by macroeconomic scenario",
        xaxis_title="Remaining month",
        yaxis_title="Cumulative PD"
    )
    st.plotly_chart(fig, use_container_width=True)

    pd12 = cumulative["Base"][min(11, remaining_term-1)]
    pdlife = cumulative["Base"][-1]
    c1,c2 = st.columns(2)
    c1.metric("12-month cumulative PD", f"{pd12:.2%}")
    c2.metric("Lifetime cumulative PD", f"{pdlife:.2%}")

    portfolio_ratio = df["current_pd"].mean() / df["orig_pd"].mean()
    if portfolio_ratio >= sicr_multiplier:
        st.warning(
            "Lifetime PD now required — significant increase in credit risk detected. "
            "The exposure set is treated as Stage 2 for this illustrative portfolio view."
        )
    else:
        st.success(
            "12-month PD is currently sufficient for Stage 1 exposures. "
            "Lifetime PD is still shown for forward-looking monitoring."
        )

# =========================================================
# 7. AASB 9 / ECL
# =========================================================
with tabs[7]:
    st.subheader("AASB 9 / IFRS 9 staging and probability-weighted ECL")

    stage1 = df[df["stage"]==1]
    stage2 = df[df["stage"]==2]
    stage3 = df[df["stage"]==3]

    # scenario ECL using aggregate portfolio and PD term structures
    disc_rate = 0.045
    discount = 1 / ((1 + disc_rate/12) ** np.arange(1, remaining_term+1))
    ead_total = df["ead"].sum()

    scenario_ecl = {}
    for name in ["Upside","Base","Downside"]:
        h = scenario_hazards[name]
        surv_prev = np.r_[1.0, np.cumprod(1-h)[:-1]]
        marginal_default = surv_prev * h

        # Stage 1 uses only first 12 months, Stage 2/3 uses remaining life
        stage1_ead = stage1["ead"].sum()
        stage2_ead = stage2["ead"].sum()
        stage3_ead = stage3["ead"].sum()

        ecl1 = stage1_ead * lgd_assumption * np.sum(marginal_default[:min(12, remaining_term)] * discount[:min(12, remaining_term)])
        ecl2 = stage2_ead * lgd_assumption * np.sum(marginal_default * discount)
        ecl3 = stage3_ead * min(0.85, lgd_assumption*1.35)
        scenario_ecl[name] = ecl1 + ecl2 + ecl3

    weighted_ecl = w_up*scenario_ecl["Upside"] + w_base*scenario_ecl["Base"] + w_down*scenario_ecl["Downside"]

    ecl_df = pd.DataFrame({
        "Scenario":["Upside","Base","Downside","Probability Weighted"],
        "ECL":[scenario_ecl["Upside"], scenario_ecl["Base"], scenario_ecl["Downside"], weighted_ecl]
    })
    ecl_df["ECL ($m)"] = ecl_df["ECL"]/1e6
    st.dataframe(ecl_df[["Scenario","ECL ($m)"]], use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Stage 1 EAD", f"${stage1['ead'].sum()/1e6:,.1f}m")
    c2.metric("Stage 2 EAD", f"${stage2['ead'].sum()/1e6:,.1f}m")
    c3.metric("Stage 3 EAD", f"${stage3['ead'].sum()/1e6:,.1f}m")
    c4.metric("Weighted ECL", f"${weighted_ecl/1e6:,.2f}m")

    stage_chart = df.groupby("stage", as_index=False)["ead"].sum()
    fig = px.bar(stage_chart, x="stage", y="ead", title="EAD by AASB 9 stage")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 8. LOSS DISTRIBUTION
# =========================================================
with tabs[8]:
    st.subheader("Forward-looking portfolio loss distribution")
    st.caption(
        "This is a synthetic economic loss simulation for risk analytics. "
        "It is related to, but not identical to, accounting ECL."
    )

    rng = np.random.default_rng(77)
    N = len(df)
    avg_ead = df["ead"].mean()
    base_pd_port = np.clip(df["current_pd"].mean(), 0.001, 0.35)

    scen_names = np.array(["Upside","Base","Downside"])
    weights = np.array([w_up, w_base, w_down])
    if weights.sum() <= 0:
        weights = np.array([0.2,0.6,0.2])
    weights = weights / weights.sum()

    scen_mult = {"Upside":0.70, "Base":1.00, "Downside":1.55 + 0.20*max(0,macro_index)}
    chosen = rng.choice(scen_names, size=n_sims, p=weights)
    systematic = rng.normal(0,1,n_sims)

    losses = np.zeros(n_sims)
    for i in range(n_sims):
        p = np.clip(base_pd_port * scen_mult[chosen[i]], 1e-5, 0.50)
        # one-factor approximation: systematic factor shifts conditional PD
        cond_pd = logistic(logit(p) + np.sqrt(asset_corr) * systematic[i])
        defaults = rng.binomial(N, cond_pd)
        lgd_draw = np.clip(rng.normal(lgd_assumption, 0.08), 0.10, 0.95)
        losses[i] = defaults * avg_ead * lgd_draw

    el = losses.mean()
    var95 = np.percentile(losses,95)
    var99 = np.percentile(losses,99)
    ul99 = var99 - el

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Expected loss", f"${el/1e6:,.2f}m")
    c2.metric("95% loss level", f"${var95/1e6:,.2f}m")
    c3.metric("99% loss level", f"${var99/1e6:,.2f}m")
    c4.metric("Unexpected loss (99%-EL)", f"${ul99/1e6:,.2f}m")

    hist = pd.DataFrame({"Loss ($m)": losses/1e6})
    fig = px.histogram(hist, x="Loss ($m)", nbins=70, title="Probability-weighted portfolio loss distribution")
    fig.add_vline(x=el/1e6, line_dash="dash", annotation_text="EL")
    fig.add_vline(x=var95/1e6, line_dash="dot", annotation_text="95%")
    fig.add_vline(x=var99/1e6, line_dash="dashdot", annotation_text="99%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scenario-specific loss distributions")
    scen_frames = []
    for scen in ["Upside","Base","Downside"]:
        p = np.clip(base_pd_port * scen_mult[scen], 1e-5, 0.50)
        sys = rng.normal(0,1,max(1500, n_sims//3))
        cond = logistic(logit(p) + np.sqrt(asset_corr)*sys)
        defaults = rng.binomial(N, cond)
        lgd_draws = np.clip(rng.normal(lgd_assumption, 0.08, len(defaults)),0.10,0.95)
        l = defaults*avg_ead*lgd_draws
        scen_frames.append(pd.DataFrame({"Loss ($m)":l/1e6,"Scenario":scen}))
    scen_df = pd.concat(scen_frames, ignore_index=True)
    fig = px.histogram(scen_df, x="Loss ($m)", color="Scenario", nbins=60, barmode="overlay",
                       opacity=0.45, title="Upside vs Base vs Downside loss distributions")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 9. MODEL MONITORING
# =========================================================
with tabs[9]:
    st.subheader("Model monitoring and drift")

    c1,c2,c3 = st.columns(3)
    c1.metric("Population Stability Index", f"{psi_value:.3f}")
    c2.metric("Average score shift", f"{df['score_current'].mean()-df['score'].mean():+.1f}")
    c3.metric("Average PD shift", f"{df['current_pd'].mean()-df['orig_pd'].mean():+.2%}")

    monitor_df = pd.DataFrame({
        "Metric":["PSI","Global AUC (synthetic)","Near-cutoff AUC (synthetic)","Calibration ratio (synthetic)"],
        "Current":[psi_value, 0.78 - 0.05*min(1,psi_value), 0.66 - 0.08*min(1,psi_value), 1.00 + 0.50*macro_index],
        "Illustrative trigger":[0.25, 0.70, 0.60, 1.20]
    })
    st.dataframe(monitor_df, use_container_width=True)

    # score distribution comparison
    sample = pd.DataFrame({
        "Development": df["score"].sample(min(2500,len(df)), random_state=2).values,
        "Current": df["score_current"].sample(min(2500,len(df)), random_state=2).values
    }).melt(var_name="Population", value_name="Score")
    fig = px.histogram(sample, x="Score", color="Population", nbins=45, barmode="overlay", opacity=0.45,
                       title="Development vs current population score distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Monitoring thresholds shown here are illustrative teaching settings, not universal regulatory thresholds."
    )

st.divider()
st.caption(
    "Important: all data and model relationships in this prototype are synthetic. "
    "The purpose is to demonstrate end-to-end credit-risk architecture and interactive scenario mechanics."
)
