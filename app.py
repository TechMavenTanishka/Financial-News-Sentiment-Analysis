import streamlit as st
import torch
import time
import pandas as pd
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Financial Sentiment AI",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: #070B1A;
    color: white;
}

.main {
    background: #070B1A;
}

.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1500px;
}

/* TITLE */

.main-title {
    font-size: 52px;
    line-height: 1.2;
    margin-top: 20px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    color: #94A3B8;
    font-size: 16px;
    margin-bottom: 25px;
}

/* PANELS */

.panel {
    background: linear-gradient(
        145deg,
        #0B1120,
        #111827
    );

    border-radius: 20px;
    padding: 25px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
    0 0 20px rgba(0,0,0,0.4);
}

/* TEXT AREA */

textarea {
    background: #09111F !important;
    color: white !important;
    border-radius: 15px !important;
    border: 1px solid #334155 !important;
    font-size: 16px !important;
}

/* BUTTON */

.stButton button {

    background: linear-gradient(
        90deg,
        #7C3AED,
        #06B6D4
    );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 12px 30px;

    font-size: 18px;

    font-weight: 600;

    transition: 0.3s;
}

.stButton button:hover {

    transform: scale(1.03);

    box-shadow:
    0 0 15px rgba(6,182,212,0.5);
}

/* RESULT CARDS */

.metric-card {

    background: #111827;

    border-radius: 15px;

    padding: 20px;

    text-align: center;

    border: 1px solid rgba(255,255,255,0.05);
}

.metric-value {

    font-size: 32px;

    font-weight: bold;

    color: #38BDF8;
}

.metric-label {

    color: #94A3B8;

    margin-top: 5px;
}

/* SENTIMENT */

.bullish {
    color: #22C55E;
    font-size: 30px;
    font-weight: bold;
}

.bearish {
    color: #EF4444;
    font-size: 30px;
    font-weight: bold;
}

.neutral {
    color: #F59E0B;
    font-size: 30px;
    font-weight: bold;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model_path = "./models/bert/finbert_model"

    tokenizer_path = "./models/bert/finbert_tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------------------------------
# LABELS
# ---------------------------------------------------

label_map = {
    0: "Bearish",
    1: "Bullish",
    2: "Neutral"
}

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

def predict_sentiment(text):

    start_time = time.time()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.nn.functional.softmax(
            outputs.logits,
            dim=-1
        )

    prediction = torch.argmax(probs).item()

    confidence = torch.max(probs).item()

    runtime = round(time.time() - start_time, 2)

    return {
        "sentiment": label_map[prediction],
        "confidence": round(confidence * 100, 2),
        "probabilities": probs.numpy()[0],
        "runtime": runtime
    }

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">📈 Financial Sentiment AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered financial article sentiment analysis using FinBERT transformers.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------

left, right = st.columns([1.3, 1])

# ---------------------------------------------------
# LEFT PANEL
# ---------------------------------------------------

with left:

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("📰 Analyze Financial Article")

    sample_cols = st.columns(4)

    samples = [
        "Apple earnings beat expectations",
        "Federal Reserve raises interest rates",
        "Tesla stock plunges after weak guidance",
        "Amazon cloud revenue jumps strongly"
    ]

    for i, sample in enumerate(samples):

        if sample_cols[i].button(sample):
            st.session_state["sample_text"] = sample

    default_text = st.session_state.get(
        "sample_text",
        ""
    )

    user_input = st.text_area(
        "",
        value=default_text,
        height=320,
        placeholder="Paste financial article, news story, tweet, earnings report..."
    )

    analyze = st.button("⚡ Analyze Article")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# RIGHT PANEL
# ---------------------------------------------------

with right:

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("📊 Analysis Results")

    if analyze and user_input.strip() != "":

        result = predict_sentiment(user_input)

        sentiment = result["sentiment"]

        confidence = result["confidence"]

        probs = result["probabilities"]

        runtime = result["runtime"]

        if sentiment == "Bullish":

            st.markdown(
                '<div class="bullish">🐂 BULLISH</div>',
                unsafe_allow_html=True
            )

        elif sentiment == "Bearish":

            st.markdown(
                '<div class="bearish">🐻 BEARISH</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="neutral">😐 NEUTRAL</div>',
                unsafe_allow_html=True
            )

        st.progress(confidence / 100)

        st.write(f"### Confidence: {confidence}%")

        st.markdown("### Input Financial Text")

        st.markdown(f"""
        <div style="
        background:#0F172A;
        padding:18px;
        border-radius:12px;
        border:1px solid rgba(255,255,255,0.08);
        color:#CBD5E1;
        line-height:1.7;
        ">
        {user_input[:500]}
        </div>
        """, unsafe_allow_html=True)

        # CHART

        chart_df = pd.DataFrame({
            "Sentiment": [
                "Bearish",
                "Bullish",
                "Neutral"
            ],
            "Probability": probs
        })

        fig = px.bar(
            chart_df,
            x="Sentiment",
            y="Probability",
            color="Sentiment",
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # METRICS

        m1, m2, m3 = st.columns(3)

        with m1:

            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{confidence}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            ''', unsafe_allow_html=True)

        with m2:

            word_count = len(user_input.split())

            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{word_count}</div>
                <div class="metric-label">Words</div>
            </div>
            ''', unsafe_allow_html=True)

        with m3:

            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{runtime}s</div>
                <div class="metric-label">Runtime</div>
            </div>
            ''', unsafe_allow_html=True)

    else:

        st.info("Awaiting financial article input...")

    st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.title("About")

st.sidebar.info("""
Financial Sentiment Analysis Dashboard
using FinBERT transformers.
""")    

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Built using FinBERT, Transformers, PyTorch, Streamlit, and Plotly."
)