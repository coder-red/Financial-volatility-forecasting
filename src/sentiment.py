import feedparser 
import pandas as pd 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime 
from config import DATA_RAW

# Load VADER 
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def fetch_daily_sentiment(ticker="SPY", keyword="market"):
    """Fetches and processes daily sentiment scores for news articles related to a specific ticker."""

    # Downloads and parses the RSS feed content from the URL
    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(rss_url)

    # Initializes an empty list to store the processed data rows
    rows = []

    # Iterates through every news article entry found in the RSS feed
    for entry in feed.entries:
        # skips the entry if the keyword isn't in the summary
        if keyword.lower() not in entry.summary.lower():
            continue

        # Converts the feed's time tuple into a standard Python date object
        published = datetime(*entry.published_parsed[:6]).date()

        # VADER sentiment analysis (replaces FinBERT)
        s = sia.polarity_scores(entry.summary)
        score = s['compound']  # -1 (negative) to +1 (positive)

        # Adds the date and the calculated score to our list of rows
        rows.append({
            "date": published,
            "sentiment": score
        })

    # Error handling: returns an empty DataFrame if no matching news was found
    if not rows:
        return pd.DataFrame(columns=["date", "sentiment"])

    # Converts the list of dictionaries into a structured Pandas DataFrame
    df = pd.DataFrame(rows)

    #  aggregate to DAILY sentiment
    # Groups all news by date and calculates the average sentiment score for that day
    daily_sentiment = (
        df.groupby("date")["sentiment"]
        .mean()
        .reset_index()
    )

    # Returns the final DataFrame containing one row per date with its mean sentiment
    return daily_sentiment
