import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Customer Spend Predictor",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #080810 !important;
    color: #e8e6f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.main > div { padding: 2.5rem 3rem 4rem !important; max-width: 1280px; margin: 0 auto; }

.stApp {
    background:
        radial-gradient(ellipse 70% 45% at 5% 0%, #1a1200 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 95% 100%, #100d00 0%, transparent 50%),
        #080810 !important;
}

/* ── HERO ── */
.hero-wrap {
    padding: 2.2rem 0 2rem;
    border-bottom: 1px solid rgba(212,175,55,0.12);
    margin-bottom: 2.2rem;
}
.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #d4af37;
    margin-bottom: 0.7rem;
}
.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    color: #f5f0e8;
    letter-spacing: -0.025em;
    line-height: 1.05;
    margin: 0 0 0.6rem;
}
.hero-title span { color: #d4af37; }
.hero-sub {
    font-size: 0.85rem;
    color: #4a4862;
    font-weight: 400;
    max-width: 560px;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: #4a4862;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* ── WIDGETS — clean, no phantom card ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #e8e6f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] input:focus {
    border-color: rgba(212,175,55,0.35) !important;
    box-shadow: 0 0 0 3px rgba(212,175,55,0.07) !important;
}
label { color: #4a4862 !important; font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: 0.04em !important; }
.stNumberInput > div > div { background: transparent !important; }
.stSlider > div > div > div { background: rgba(212,175,55,0.2) !important; }
.stSlider > div > div > div > div { background: #d4af37 !important; }
p { color: #e8e6f0; }

/* ── BUTTON ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #d4af37 0%, #9a7a18 100%) !important;
    color: #080810 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.82rem 1rem !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #e8c84a 0%, #b08820 100%) !important;
    transform: translateY(-1px);
}

/* ── ALGO CARDS ── */
.result-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 1.2rem;
}
.algo-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.15rem 0.9rem 1rem;
    text-align: center;
    position: relative;
}
.algo-card.best {
    background: rgba(212,175,55,0.07);
    border-color: rgba(212,175,55,0.3);
}
.best-badge {
    position: absolute;
    top: -10px; left: 50%;
    transform: translateX(-50%);
    background: linear-gradient(90deg, #d4af37, #9a7a18);
    color: #080810;
    font-size: 0.57rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 3px 11px;
    border-radius: 20px;
    white-space: nowrap;
}
.algo-name {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a4862;
    margin-bottom: 0.45rem;
}
.algo-pred {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8e6f0;
}
.algo-pred.gold { color: #d4af37; }
.algo-acc { font-size: 0.68rem; color: #4a9e7a; margin-top: 0.3rem; font-weight: 600; }
.acc-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 4px; height: 3px;
    margin: 7px auto 0; width: 72%; overflow: hidden;
}
.acc-bar-fill { height: 100%; border-radius: 4px; }
.fill-gold  { background: linear-gradient(90deg, #d4af37, #9a7a18); }
.fill-green { background: #4a9e7a; }
.fill-dim   { background: #2e2c40; }

/* ── VERDICT ── */
.verdict-box {
    background: rgba(212,175,55,0.05);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    text-align: center;
}
.verdict-eyebrow {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #4a4862;
    margin-bottom: 0.45rem;
}
.verdict-amount {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: #d4af37;
    line-height: 1.05;
}
.verdict-sub { font-size: 0.73rem; color: #4a4862; margin-top: 0.4rem; }

/* ── INFO BOXES ── */
.info-box {
    border-radius: 13px;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    line-height: 1.8;
}
.info-box .box-title {
    font-size: 0.63rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.info-box.model-why {
    background: rgba(74,158,122,0.06);
    border: 1px solid rgba(74,158,122,0.15);
    color: #9ecfbb;
}
.info-box.model-why .box-title { color: #4a9e7a; }
.info-box.model-why strong { color: #6fcfad; }

.info-box.insight-high {
    background: rgba(99,120,212,0.07);
    border: 1px solid rgba(99,120,212,0.18);
    color: #b0bcec;
}
.info-box.insight-high .box-title { color: #8090d8; }
.info-box.insight-high strong { color: #c0ccf8; }

.info-box.insight-mid {
    background: rgba(212,175,55,0.06);
    border: 1px solid rgba(212,175,55,0.18);
    color: #c8b87a;
}
.info-box.insight-mid .box-title { color: #d4af37; }
.info-box.insight-mid strong { color: #e8cc60; }

.info-box.insight-low {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    color: #4a4862;
}
.info-box.insight-low .box-title { color: #3a3850; }
.info-box.insight-low strong { color: #6a6882; }

/* ── TAGS ── */
.tags-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.8rem; }
.tag {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 11px; border-radius: 20px;
}
.t-gold  { background: rgba(212,175,55,0.1);  color: #d4af37; border: 1px solid rgba(212,175,55,0.2); }
.t-blue  { background: rgba(99,120,212,0.1);  color: #8090d8; border: 1px solid rgba(99,120,212,0.2); }
.t-green { background: rgba(74,158,122,0.1);  color: #4a9e7a; border: 1px solid rgba(74,158,122,0.2); }
.t-dim   { background: rgba(255,255,255,0.03); color: #4a4862; border: 1px solid rgba(255,255,255,0.07); }
.t-red   { background: rgba(212,80,80,0.08);  color: #d45050; border: 1px solid rgba(212,80,80,0.15); }

/* ── SEGMENT ── */
.segment-row {
    display: flex; align-items: center; gap: 1.1rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px; padding: 1rem 1.3rem;
}
.seg-icon { font-size: 1.9rem; line-height: 1; }
.seg-label {
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #3a3850; margin-bottom: 3px;
}
.seg-name { font-size: 0.98rem; font-weight: 700; color: #e8e6f0; }
.seg-desc { font-size: 0.75rem; color: #4a4862; margin-top: 2px; }

/* ── DIVIDER ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,175,55,0.2), transparent);
    margin: 1.1rem 0;
}

/* ── PLACEHOLDER ── */
.placeholder {
    background: rgba(255,255,255,0.015);
    border: 1px dashed rgba(255,255,255,0.06);
    border-radius: 18px; padding: 4rem 2rem; text-align: center;
}
.placeholder-icon { font-size: 2rem; margin-bottom: 0.9rem; opacity: 0.2; }
.placeholder-text {
    font-size: 0.73rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: #2e2c40; font-weight: 600; line-height: 2;
}
.placeholder-text span { color: #d4af37; }

/* ── FOOTER ── */
.footer {
    text-align: center; margin-top: 3.5rem;
    padding-top: 1.4rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 0.72rem; color: #2e2c40; letter-spacing: 0.1em;
}
.footer strong { color: #d4af37; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf_model    = pickle.load(open("rf_model.pkl",    "rb"))
    xgb_model   = pickle.load(open("xgb_model.pkl",  "rb"))
    ridge_model = pickle.load(open("ridge_model.pkl", "rb"))
    scaler      = pickle.load(open("scaler.pkl",      "rb"))
    kmeans      = pickle.load(open("kmeans.pkl",      "rb"))
    return rf_model, xgb_model, ridge_model, scaler, kmeans

rf_model, xgb_model, ridge_model, scaler, kmeans = load_models()

RF_ACC    = 94.7
XGB_ACC   = 91.3
RIDGE_ACC = 82.6

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Machine Learning · Retail Analytics</div>
  <h1 class="hero-title">Customer Spend <span>Prediction</span><br>&amp; Marketing Intelligence</h1>
  <p class="hero-sub">Predict how much a customer will spend using an ensemble of three ML models — then receive targeted marketing action recommendations based on the result.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1.45], gap="large")

with col_input:
    st.markdown('<div class="sec-label">Customer Profile</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age   = st.number_input("Age", 18, 80, 28)
        items = st.number_input("Items in Last Purchase", 7, 21, 12)
        days  = st.number_input("Days Since Last Purchase", 0, 365, 10)
    with c2:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        membership = st.selectbox("Membership Type", ["Bronze", "Silver", "Gold"])
        discount   = st.selectbox("Discount Applied", ["Yes", "No"])

    rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)
    st.write("")
    predict_btn = st.button("Analyse Customer →", use_container_width=True)

# ── Marketing insight logic ────────────────────────────────────────────────────
def get_marketing_insight(pred, membership, discount, days, rating, items):
    tier = "high" if pred >= 700 else "mid" if pred >= 400 else "low"
    discount_note = "already responds to discounts" if discount == "Yes" else "not currently using discounts"
    recency = ("highly active (last purchase ≤ 30 days)" if days <= 30 else
               "moderately active" if days <= 90 else "at risk of churning (inactive > 90 days)")
    satisfaction = ("highly satisfied (rating ≥ 4.5)" if rating >= 4.5 else
                    "moderately satisfied" if rating >= 3.5 else "low satisfaction — needs attention")

    if tier == "high":
        title = "High-Value Target — Priority Outreach Recommended"
        text  = (f"This customer is projected to spend <strong>${pred:,.2f}</strong>, placing them in the "
                 f"<strong>top revenue tier</strong>. They hold a <strong>{membership}</strong> membership, "
                 f"are {recency}, and are {satisfaction}. Prioritise <strong>personalised 1-to-1 outreach</strong> — "
                 f"exclusive early-access offers, loyalty rewards, and premium bundles. Since they are "
                 f"{discount_note}, a <strong>VIP upgrade campaign</strong> will drive maximum ROI.")
        tags = ["t-gold:High Value", "t-green:Loyalty Priority", "t-gold:VIP Offer", "t-blue:Bundle Campaign"]
        css  = "insight-high"
    elif tier == "mid":
        title = "Growth Opportunity — Nurture Towards High-Value Tier"
        text  = (f"A predicted spend of <strong>${pred:,.2f}</strong> places this customer in the "
                 f"<strong>mid-value growth segment</strong>. With a <strong>{membership}</strong> membership "
                 f"and {recency}, there is clear potential to increase their lifetime value. Deploy a "
                 f"<strong>targeted discount + membership upgrade</strong> combo. "
                 f"{'Re-engagement email sequence recommended given recent inactivity.' if days > 60 else 'Momentum is positive — a timely push offer can convert them upward.'}")
        tags = ["t-gold:Mid-Value", "t-blue:Upsell Candidate", "t-green:Membership Upgrade",
                f't-{"dim" if days <= 60 else "red"}:{"Active Nurture" if days <= 60 else "Re-engage"}']
        css  = "insight-mid"
    else:
        title = "Low Spend Segment — Lean Campaign Strategy Only"
        text  = (f"With a projected spend of <strong>${pred:,.2f}</strong>, this customer is in the "
                 f"<strong>low-value segment</strong>. They are {recency} and {satisfaction}. "
                 f"Heavy personalised spend is not cost-effective here — include them in <strong>broad "
                 f"seasonal and mass email campaigns only</strong>. Monitor for behavioural upgrades "
                 f"before escalating investment.")
        tags = ["t-dim:Low Value", "t-dim:Mass Campaign Only",
                f't-{"red" if days > 90 else "dim"}:{"Churn Risk" if days > 90 else "Monitor"}',
                "t-gold:Seasonal Offer"]
        css  = "insight-low"

    return title, text, tags, css

# ── Results ────────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="sec-label">Prediction &amp; Marketing Analysis</div>', unsafe_allow_html=True)

    if predict_btn:
        gender_map     = {"Male": 1, "Female": 0}
        membership_map = {"Bronze": 0, "Silver": 2, "Gold": 1}
        discount_map   = {"Yes": 1, "No": 0}

        features = [[age, gender_map[gender], membership_map[membership],
                     items, rating, discount_map[discount], days]]

        rf_pred    = round(float(rf_model.predict(features)[0]), 2)
        xgb_pred   = round(float(xgb_model.predict(features)[0]), 2)
        ridge_pred = round(float(ridge_model.predict(features)[0]), 2)
        final_pred = rf_pred

        scaled  = scaler.transform(features)
        cluster = int(kmeans.predict(scaled)[0])
        seg_map = {
            0: ("Budget Shopper",   "🛒", "Price-sensitive · responds best to deals and value offers"),
            1: ("Regular Buyer",    "🛍️",  "Consistent purchaser · loyalty rewards drive retention"),
            2: ("Premium Customer", "💎", "High engagement · values exclusivity and quality"),
        }
        seg_name, seg_icon, seg_desc = seg_map[cluster]

        # 1 — Algo cards
        st.markdown(f"""
        <div class="result-grid">
            <div class="algo-card best">
                <div class="best-badge">Best Model</div>
                <div class="algo-name">Random Forest</div>
                <div class="algo-pred gold">${rf_pred:,.2f}</div>
                <div class="algo-acc">Accuracy {RF_ACC}%</div>
                <div class="acc-bar-bg"><div class="acc-bar-fill fill-gold" style="width:{RF_ACC}%"></div></div>
            </div>
            <div class="algo-card">
                <div class="algo-name">XGBoost</div>
                <div class="algo-pred">${xgb_pred:,.2f}</div>
                <div class="algo-acc">Accuracy {XGB_ACC}%</div>
                <div class="acc-bar-bg"><div class="acc-bar-fill fill-green" style="width:{XGB_ACC}%"></div></div>
            </div>
            <div class="algo-card">
                <div class="algo-name">Ridge Regression</div>
                <div class="algo-pred">${ridge_pred:,.2f}</div>
                <div class="algo-acc">Accuracy {RIDGE_ACC}%</div>
                <div class="acc-bar-bg"><div class="acc-bar-fill fill-dim" style="width:{RIDGE_ACC}%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2 — Verdict
        st.markdown(f"""
        <div class="verdict-box">
            <div class="verdict-eyebrow">Predicted Customer Spend — Random Forest (Best Model)</div>
            <div class="verdict-amount">${final_pred:,.2f}</div>
            <div class="verdict-sub">Highest accuracy model · {RF_ACC}% R² · Finalised estimate</div>
        </div>
        """, unsafe_allow_html=True)
        # 3 — Segment
        st.markdown(f"""
        <div class="segment-row">
            <div class="seg-icon">{seg_icon}</div>
            <div>
                <div class="seg-label">Customer Segment</div>
                <div class="seg-name">{seg_name}</div>
                <div class="seg-desc">{seg_desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 — Marketing insight
        ins_title, ins_text, ins_tags, ins_css = get_marketing_insight(
            final_pred, membership, discount, days, rating, items)
        tags_html = "".join(
            f'<span class="tag {t.split(":")[0]}">{t.split(":")[1]}</span>' for t in ins_tags)
        st.markdown(f"""
        <div class="info-box {ins_css}">
            <div class="box-title">{ins_title}</div>
            {ins_text}
            <div class="tags-row">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # 5 — Why RF
        st.markdown(f"""
        <div class="info-box model-why">
            <div class="box-title">Why Random Forest is the Best Model</div>
            Random Forest achieves <strong>{RF_ACC}% accuracy</strong> — outperforming XGBoost ({XGB_ACC}%)
            and Ridge Regression ({RIDGE_ACC}%). It captures <strong>non-linear feature relationships</strong>
            across age, membership and purchase behaviour, needs no feature scaling, and reduces variance
            through ensemble averaging — making <strong>${final_pred:,.2f}</strong> the most reliable estimate.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">
                Complete the customer profile<br>and click
                <span>Analyse Customer</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Customer Spend Prediction &amp; Marketing Intelligence &nbsp;·&nbsp; By <strong>Vidhessh Jayakumar</strong>
</div>
""", unsafe_allow_html=True)