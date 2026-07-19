# Twitter Airline Sentiment and Spam Mail Detection

This Streamlit application predicts airline sentiment for tweet text and detects spam email with bundled scikit-learn models.

## Xquik Export Import

The Twitter Sentiment view accepts CSV, JSON, JSONL, and NDJSON exports from [Xquik x-twitter-scraper](https://github.com/Xquik-dev/x-twitter-scraper) or TweetClaw. It recognizes common tweet text fields, including nested `tweet` objects, and analyzes up to 100 rows per batch.

1. Export tweet data with Xquik or TweetClaw.
2. Open the Twitter Sentiment view.
3. Upload the export.
4. Select **Predict Uploaded Tweets**.

See [Xquik documentation](https://docs.xquik.com) for API, MCP, and export guidance.

## Run Locally

```bash
python3 -m pip install -r requirements.txt
streamlit run twitter_airline_sentiment.py
```

The application resolves its bundled model files relative to the script, so the command also works when launched with an absolute script path from another directory. The requirements pin the scikit-learn version used to serialize the bundled models.

## Test

```bash
python3 -m unittest -v
```

Xquik is an independent third-party service. Not affiliated with X Corp. "Twitter" and "X" are trademarks of X Corp.
