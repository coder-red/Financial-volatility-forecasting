import feedparser
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from datetime import date, datetime
from pytrends.request import TrendReq
from config import DATA_RAW, VOL_WINDOW
import time
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Load FinBERT (ONNX)
# -----------------------------
MODEL_NAME = "ProsusAI/finbert"
ONNX_MODEL_PATH = "../finbert.onnx"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
session = InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

LABELS = ["negative", "neutral", "positive"]


def finbert_onnx_sentiment(text: str) -> float:
    """
    Run FinBERT sentiment using ONNX.
    Returns signed sentiment score in [-1, 1].
    """
    inputs = tokenizer(
        text, 
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=128,
        return_token_type_ids=True, 
    )


    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64),
        },
    )



    logits = outputs[0][0]
    logits = logits - np.max(logits)          # for numerical stability
    probs = np.exp(logits) / np.sum(np.exp(logits))

    label_idx = int(np.argmax(probs))
    score = float(probs[label_idx])

    if LABELS[label_idx] == "negative":
        return -score
    elif LABELS[label_idx] == "positive":
        return score
    else:
        return 0.0


# -----------------------------
# MULTI-SOURCE NEWS SENTIMENT
# -----------------------------
def fetch_daily_sentiment_multi_source(ticker="SPY", keyword="market"):
    """
    Aggregate sentiment from multiple RSS feeds using FinBERT-ONNX.
    """
    rss_urls = [
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.ft.com/markets?format=rss",
        "https://www.wsj.com/xml/rss/3_7085.xml",
    ]

    all_rows = []

    def fetch_single_feed(rss_url):
        rows = []
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                text = entry.get("summary", entry.get("description", ""))
                if keyword.lower() not in text.lower():
                    continue
                try:
                    published = datetime(*entry.published_parsed[:6]).date()
                except Exception:
                    continue

                sentiment = finbert_onnx_sentiment(text)
                rows.append(
                    {
                        "date": published,
                        "sentiment": sentiment,
                        "source": rss_url.split("/")[2],
                    }
                )
        except Exception as e:
            print(f"⚠️ Failed RSS fetch: {rss_url} | {e}")
        return rows

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_single_feed, rss_urls)

    for r in results:
        all_rows.extend(r)

    if not all_rows:
        return pd.DataFrame(columns=["date", "sentiment"])

    df = pd.DataFrame(all_rows)
    daily_sentiment = df.groupby("date")["sentiment"].mean().reset_index()

    print(f"✅ {len(df)} headlines | {df['source'].nunique()} sources")
    return daily_sentiment


# -----------------------------
# GOOGLE TRENDS SENTIMENT
# -----------------------------
def fetch_google_trends():
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    keywords = ["stock market", "market volatility"]

    try:
        pytrends.build_payload(keywords, timeframe="today 3-m")
        time.sleep(2)
        df = pytrends.interest_over_time()

        if df.empty or "isPartial" not in df.columns:
            return pd.DataFrame(columns=["date", "sentiment"])

        df["trends"] = df[keywords].mean(axis=1)
        df["sentiment"] = 2 * (
            (df["trends"] - df["trends"].min())
            / (df["trends"].max() - df["trends"].min())
        ) - 1

        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df = (
            df[["date", "sentiment"]]
            .set_index("date")
            .resample("D")
            .ffill()
            .reset_index()
        )
        df["date"] = df["date"].dt.date

        return df

    except Exception as e:
        print(f"❌ Google Trends failed: {e}")
        return pd.DataFrame(columns=["date", "sentiment"])



# -----------------------------
# CREATE DAILY VOLATILITY
# -----------------------------

def create_daily_vol(ticker="SPY", window=VOL_WINDOW):
    """
    Create daily volatility CSV from historical price data.
    This is done to quantify recent market risk for volatility-weighted sentiment.
    Saves to DATA_RAW / 'daily_vol.csv'.
    """
    price_path = DATA_RAW / f"{ticker}.csv"
    if not price_path.exists():
        print(f"❌ {ticker}.csv not found in DATA_RAW. Can't create daily_vol.csv")
        return None

    prices = pd.read_csv(price_path)
    prices["date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("date")

    # daily returns
    prices["returns"] = prices["Close"].pct_change()

    # rolling volatility
    prices["vol"] = prices["returns"].rolling(window).std() # Calculates the 20 day rolling standard deviation of daily returns to quantify recent market risk.

    daily_vol = prices[["date", "vol"]].dropna()
    daily_vol.to_csv(DATA_RAW / "daily_vol.csv", index=False)
    print(f"✅ daily_vol.csv created ({len(daily_vol)} days)")
    return daily_vol



# -----------------------------
# VOLATILITY-WEIGHTED SENTIMENT
# -----------------------------
def apply_volatility_weighting(sentiment_df, vol_df):
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    vol_df["date"] = pd.to_datetime(vol_df["date"]).dt.date

    df = sentiment_df.merge(vol_df, on="date", how="left") #join sentiment with volatility on date
    df["vol_z"] = (df["vol"] - df["vol"].mean()) / df["vol"].std() # calculate z-score(How many std away from the average) of volatility
    df["sentiment_vol"] = df["sentiment"] * (1 + df["vol_z"])
    df["sentiment_vol"] = df["sentiment_vol"].fillna(df["sentiment"]) # fill missing vol-weighted sentiment with original sentiment

    return df[["date", "sentiment_vol"]]


# -----------------------------
# COMBINE SENTIMENT SOURCES
# -----------------------------
def combine_sentiment_sources(vol_weighted=True):
    news = pd.read_csv(DATA_RAW / "news_sentiment.csv")
    trends = pd.read_csv(DATA_RAW / "google_trends.csv")

    news["date"] = pd.to_datetime(news["date"]).dt.date
    trends["date"] = pd.to_datetime(trends["date"]).dt.date

    combined = news.merge(
        trends, on="date", how="outer", suffixes=("_news", "_trends")
    ).sort_values("date") 

    # fill missing news and trends sentiment
    combined["sentiment_news"] = combined["sentiment_news"].ffill().bfill().fillna(0) 
    combined["sentiment_trends"] = combined["sentiment_trends"].ffill().bfill().fillna(0)

    combined["sentiment"] = (
        0.7 * combined["sentiment_news"] + 0.3 * combined["sentiment_trends"]
    ) # weighted average:  professional news (70%) is more reliable than trends (30%)

    if vol_weighted:
        vol_path = DATA_RAW / "daily_vol.csv"
        if vol_path.exists():
            vol = pd.read_csv(vol_path)
            combined = apply_volatility_weighting(
                combined[["date", "sentiment"]], vol
            )
            combined.rename(columns={"sentiment_vol": "sentiment"}, inplace=True)
        else:
            print("⚠️ daily_vol.csv missing — skipping volatility weighting")


    combined.to_csv(DATA_RAW / "combined_sentiment.csv", index=False)
    print(f"✅ Combined sentiment saved: {len(combined)} days")

    return combined
