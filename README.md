# Financial News Sentiment Analysis

AI-powered Financial Sentiment Analysis system using Deep Learning and Transformer-based NLP models.

---

## Project Overview

This project analyzes finance-related news headlines, tweets, and articles to classify sentiment into:

- 🐻 Bearish
- 😐 Neutral
- 🐂 Bullish

The project compares traditional Deep Learning architectures with Transformer-based models for financial sentiment classification.

Final deployment is built using Streamlit.

---

## Models Implemented

### Deep Learning Models
- RNN
- LSTM
- GRU

### Transformer Models
- BERT
- FinBERT

---

## Features

- Financial sentiment classification
- Real-time prediction system
- Bearish / Neutral / Bullish detection
- Interactive Streamlit dashboard
- Confidence score visualization
- Transformer-based NLP pipeline
- Financial-domain sentiment analysis

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Plotly
- NLTK

---

## Dataset

Finance-related tweets and news headlines labeled into:

- Bearish
- Neutral
- Bullish

---

## Final Selected Model

FinBERT achieved the best overall performance because it is specifically trained on financial-domain text and understands financial vocabulary more effectively than generic NLP models.

---

## Project Workflow

```text
Raw Dataset
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Model Training
(RNN / LSTM / GRU / BERT / FinBERT)
    ↓
Model Evaluation
    ↓
Model Comparison
    ↓
Prediction Pipeline
    ↓
Streamlit Deployment
