import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(cleaned_text, models_dict):
    """Performs sentiment analysis using FinBERT, VADER, and LM."""
    sentences = nltk.sent_tokenize(cleaned_text)
    words = nltk.word_tokenize(cleaned_text.lower())

    # FinBERT Analysis
    finbert_scores = models_dict["finbert"](sentences)
    finbert_pos = sum(1 for s in finbert_scores if s['label'] == 'positive')
    finbert_neg = sum(1 for s in finbert_scores if s['label'] == 'negative')
    finbert_neu = len(sentences) - finbert_pos - finbert_neg
    finbert_avg = (finbert_pos - finbert_neg) / len(sentences) if sentences else 0

    # VADER Analysis
    vader_scores = [models_dict["vader"].polarity_scores(s)['compound'] for s in sentences]
    vader_pos = sum(1 for s in vader_scores if s > 0.05)
    vader_neg = sum(1 for s in vader_scores if s < -0.05)
    vader_neu = len(vader_scores) - vader_pos - vader_neg
    vader_avg = sum(vader_scores) / len(vader_scores) if vader_scores else 0

    # Loughran-McDonald Analysis
    lm_pos = sum(1 for w in words if w in models_dict["lm"]["positive"])
    lm_neg = sum(1 for w in words if w in models_dict["lm"]["negative"])
    lm_unc = sum(1 for w in words if w in models_dict["lm"]["uncertainty"])
    lm_total_sentiment = lm_pos + lm_neg
    lm_score = (lm_pos - lm_neg) / lm_total_sentiment if lm_total_sentiment > 0 else 0

    return {
        'FinBERT': {'score': finbert_avg, 'counts': {'positive': finbert_pos, 'negative': finbert_neg, 'neutral': finbert_neu}},
        'VADER': {'score': vader_avg, 'counts': {'positive': vader_pos, 'negative': vader_neg, 'neutral': vader_neu}},
        'LM': {'score': lm_score, 'counts': {'positive': lm_pos, 'negative': lm_neg, 'uncertainty': lm_unc}}
    }
