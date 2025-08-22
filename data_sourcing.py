import yfinance as yf
import requests
from datetime import datetime, timedelta
import streamlit as st

def handle_api_request(url, payload, headers):
    """Handles API requests with basic error handling."""
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def clean_text_api(transcript, api_key):
    """Cleans transcript text using the xAI API."""
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = (
        "Please remove all operator instructions, legal disclaimers, metadata, and introductory pleasantries "
        "from the following earnings call transcript. Return only the core content, which includes the "
        "prepared remarks from executives and the question-and-answer (Q&A ) session. The output should be clean, "
        f"continuous text.\n\n---\n\n{transcript}"
    )
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}]}

    data = handle_api_request(url, payload, headers)
    if data and data.get("choices"):
        return data["choices"][0]["message"]["content"]
    st.error("Failed to clean text using API. Using original transcript.")
    return transcript

def get_market_performance(ticker, call_date):
    """Fetches market performance for the 7 days following the call date."""
    try:
        stock = yf.Ticker(ticker)
        start_date = datetime.combine(call_date, datetime.min.time())
        end_date = start_date + timedelta(days=8)

        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if len(hist) < 2:
            st.warning(f"Not enough market data found for {ticker} after {start_date.date()}. Need at least 2 trading days.")
            return 0.0, None

        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]

        market_return = ((end_price - start_price) / start_price) * 100
        return market_return, hist
    except Exception as e:
        st.error(f"Failed to fetch market data for {ticker}: {e}")
        return 0.0, None
