import streamlit as st
import pandas as pd
import nltk
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import traceback

# Import functions from our other modules
from data_sourcing import clean_text_api, get_market_performance
from analysis import analyze_sentiment
from visualizations import create_visualizations
from reporting import generate_report_api, create_pdf_report

# --- Configuration and Setup ---
st.set_page_config(page_title="ECSA Tool", layout="wide")

# --- API KEY (Hardcoded as requested) ---
API_KEY = "xai-6jSdlT02NVgAZDylmMPOfHVLLFH1xTugFfqRCfVUlYHxj1Yt1kFmBs1Jex2R7rT6opSyoH7GvvuKYEL2"

# Caching models and data for performance
@st.cache_resource
def load_models_and_data():
    """Loads all necessary models and data files once."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

        lm_dict = pd.read_csv("LMMD.csv")

        lm_positive_words = set(lm_dict[lm_dict['Positive'] > 0]['Word'].str.lower())
        lm_negative_words = set(lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower())
        lm_uncertainty_words = set(lm_dict[lm_dict['Uncertainty'] > 0]['Word'].str.lower())

        sentiment_analyzer = SentimentIntensityAnalyzer()
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

        return {
            "vader": sentiment_analyzer,
            "finbert": finbert,
            "lm": {
                "positive": lm_positive_words,
                "negative": lm_negative_words,
                "uncertainty": lm_uncertainty_words
            }
        }
    except FileNotFoundError:
        st.error("Loughran-McDonald dictionary (LMMD.csv) not found. Please make sure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None

models = load_models_and_data()

# --- Streamlit User Interface ---
st.title("📈 Earnings Call Sentiment Analyzer (ECSA)")
st.markdown("Upload an earnings call transcript, provide the company ticker and call date, and get a full sentiment and market analysis report.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input Data")
    uploaded_file = st.file_uploader("Upload Earnings Call Transcript (.txt)", type="txt")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")
    call_date = st.date_input("Enter Earnings Call Date", datetime.now() - timedelta(days=7))

with col2:
    st.subheader("2. Analysis Control")
    st.write("Click the button below to start the analysis.")
    analyze_button = st.button("🚀 Analyze and Generate Report", type="primary")

# --- Main Analysis Workflow ---
if analyze_button:
    if not uploaded_file or not ticker or not call_date:
        st.error("Please provide all inputs: a transcript file, a ticker, and a date.")
    elif not models:
         st.error("Models could not be loaded. Please check the console for errors and ensure LMMD.csv is present.")
    else:
        with st.spinner("Analyzing... This may take a few minutes."):
            try:
                transcript = uploaded_file.read().decode("utf-8")

                st.info("✅ **Step 1: Cleansing transcript...**")
                cleaned_text = clean_text_api(transcript, API_KEY)

                st.info("✅ **Step 2: Performing sentiment analysis...**")
                sentiment_results = analyze_sentiment(cleaned_text, models)

                st.info("✅ **Step 3: Fetching market data...**")
                market_change, market_history = get_market_performance(ticker, call_date)

                st.info("✅ **Step 4: Generating visualizations...**")
                figs = create_visualizations(sentiment_results, market_history, ticker, cleaned_text)

                st.info("✅ **Step 5: Generating AI-powered report...**")
                report_text = generate_report_api(cleaned_text, sentiment_results, market_change, API_KEY)

                st.info("✅ **Step 6: Compiling PDF document...**")
                pdf_buffer = create_pdf_report(report_text, figs)

                st.success("Analysis Complete!")

                st.subheader("📊 Analysis Results")
                if 'word_cloud' in figs:
                    st.pyplot(figs['word_cloud'])
                if 'score_comparison' in figs:
                    st.pyplot(figs['score_comparison'])
                if 'market_performance' in figs:
                    st.pyplot(figs['market_performance'])

                st.subheader("📝 Generated Report Summary")
                st.markdown(report_text, unsafe_allow_html=True)

                st.download_button(
                    label="📥 Download Full PDF Report",
                    data=pdf_buffer,
                    file_name=f"{ticker}_ECSA_Report_{call_date}.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.code(traceback.format_exc())
