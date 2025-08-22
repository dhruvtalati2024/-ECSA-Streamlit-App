import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def create_visualizations(sentiment_results, market_history, ticker, cleaned_text):
    """Creates and returns a dictionary of Matplotlib figures, including a word cloud."""
    figs = {}
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Figure 1: Sentiment Score Comparison ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    methods = ['FinBERT', 'VADER', 'LM']
    scores = [sentiment_results[m]['score'] for m in methods]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    ax1.bar(methods, scores, color=colors)
    ax1.set_title('Normalized Sentiment Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sentiment Score (-1 to 1)')
    ax1.axhline(0, color='grey', linewidth=0.8)
    plt.tight_layout()
    figs['score_comparison'] = fig1

    # --- Figure 2: Market Performance Chart ---
    if market_history is not None and not market_history.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        market_history['Close'].plot(ax=ax2, marker='o', linestyle='-')
        ax2.set_title(f'{ticker} Stock Price: 7-Day Post-Call Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Closing Price (USD)')
        ax2.set_xlabel('Date')
        plt.tight_layout()
        figs['market_performance'] = fig2

    # --- Figure 3: Word Cloud ---
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        "company", "quarter", "earnings", "call", "revenue", "year", "billion",
        "million", "financial", "results", "conference", "operator", "question",
        "analyst", "thank", "thanks", "please", "good", "morning", "afternoon"
    ])

    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        stopwords=custom_stopwords, collocations=False
    ).generate(cleaned_text)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.set_title('Key Topics Word Cloud', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.tight_layout()
    figs['word_cloud'] = fig3

    return figs
