## Project Overview

This project performs financial sentiment analysis on finance-related
news headlines, tweets, and articles using Deep Learning and Transformer models.

The system classifies text into:
- Bearish
- Neutral
- Bullish

Models Implemented:
- RNN
- LSTM
- GRU
- BERT
- FinBERT

Final deployment is built using Streamlit.

# Financial News Sentiment Analysis

AI-powered financial sentiment analysis system using Deep Learning and FinBERT.

## Features

- Financial sentiment classification
- Bearish / Neutral / Bullish prediction
- Deep Learning Models:
  - RNN
  - LSTM
  - GRU
  - BERT
  - FinBERT
- Interactive Streamlit Dashboard
- Real-time prediction
- Confidence visualization

## Tech Stack

- Python
- PyTorch
- Transformers
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Plotly

## Dataset

Finance-related tweets/news headlines labeled as:
- Bearish
- Neutral
- Bullish

## Final Selected Model

FinBERT achieved the best performance among all models.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

---

# STEP 5 — CHECK REQUIREMENTS FILE

Run:

```bash id="5x9x0g"
pip freeze > requirements.txt
