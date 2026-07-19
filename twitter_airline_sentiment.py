import csv
import io
import json
import pickle
from pathlib import Path

import streamlit as st

APP_DIRECTORY = Path(__file__).resolve().parent
XQUIK_REPOSITORY_URL = "https://github.com/Xquik-dev/x-twitter-scraper"
NORMALIZED_TWEET_TEXT_FIELDS = (
    "text",
    "full_text",
    "tweet_text",
    "tweettext",
    "content",
    "clean_text",
)
TWEET_LIST_FIELDS = ("tweets", "data", "items", "results")
BATCH_LIMIT = 100
SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def load_pickle_artifact(file_name):
    with (APP_DIRECTORY / file_name).open("rb") as artifact_file:
        return pickle.load(artifact_file)


@st.cache_resource
def load_models():
    return (
        load_pickle_artifact("airline_vectorizer.pkl"),
        load_pickle_artifact("twitter_sentiment_pred.sav"),
        load_pickle_artifact("spam_mail_predict.sav"),
        load_pickle_artifact("spam_vectorizer.pkl"),
    )


(
    airline_vectorizer,
    sentiment_model,
    spam_model,
    spam_vectorizer,
) = load_models()


def predict_twitter_sentiment(new_tweet):
    X_new = airline_vectorizer.transform([new_tweet])
    prediction = sentiment_model.predict(X_new)[0]
    return SENTIMENT_LABELS.get(prediction, str(prediction))


def parse_xquik_export(file_name, file_bytes):
    text = file_bytes.decode("utf-8-sig")
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        return parse_xquik_csv(text)
    if lower_name.endswith((".jsonl", ".ndjson")):
        return parse_xquik_json_lines(text)
    return parse_xquik_json(text)


def parse_xquik_csv(text):
    rows = csv.DictReader(io.StringIO(text))
    tweets = []
    for row in rows:
        tweet_text = extract_tweet_text(row)
        if tweet_text:
            tweets.append(tweet_text)
    if not tweets:
        raise ValueError("No tweet text column found in the CSV export.")
    return tweets


def parse_xquik_json(text):
    stripped = text.strip()
    if not stripped:
        raise ValueError("The Xquik export is empty.")

    payload = json.loads(stripped)
    records = records_from_payload(payload)
    return extract_tweets_from_records(records, "JSON")


def parse_xquik_json_lines(text):
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return extract_tweets_from_records(records, "JSONL")


def extract_tweets_from_records(records, label):
    tweets = []
    for record in records:
        tweet_text = extract_tweet_text(record)
        if tweet_text:
            tweets.append(tweet_text)
    if not tweets:
        raise ValueError("No tweet text field found in the " + label + " export.")
    return tweets


def records_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for field in TWEET_LIST_FIELDS:
            records = payload.get(field)
            if isinstance(records, list):
                return records
        return [payload]
    return []


def extract_tweet_text(record):
    if not isinstance(record, dict):
        return ""

    normalized_record = {
        str(field).strip().lower().replace(" ", "_").replace("-", "_"): value
        for field, value in record.items()
    }
    for field in NORMALIZED_TWEET_TEXT_FIELDS:
        value = normalized_record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    tweet = normalized_record.get("tweet")
    if isinstance(tweet, dict):
        return extract_tweet_text(tweet)
    return ""


def predict_spam_mail(new_email):
    X_new = spam_vectorizer.transform([new_email])
    prediction = spam_model.predict(X_new)[0]

    if prediction == 0:
        return "Not Spam"
    if prediction == 1:
        return "Spam"
    return str(prediction)


def render_batch_import():
    uploaded_file = st.file_uploader(
        "Upload Xquik or TweetClaw export",
        type=["csv", "json", "jsonl", "ndjson"],
    )
    if uploaded_file is None:
        return

    try:
        tweets = parse_xquik_export(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        st.error("Could not read Xquik export: " + str(exc))
        return

    st.write("Loaded " + str(len(tweets)) + " tweet rows.")
    if not st.button("Predict Uploaded Tweets"):
        return

    results = [
        {
            "tweet": tweet,
            "sentiment": predict_twitter_sentiment(tweet),
        }
        for tweet in tweets[:BATCH_LIMIT]
    ]
    st.dataframe(results, use_container_width=True)
    if len(tweets) > BATCH_LIMIT:
        st.info(
            "Showing the first "
            + str(BATCH_LIMIT)
            + " rows to keep prediction responsive."
        )


def render_twitter_sentiment():
    st.markdown(
        f"Analyze exports from [Xquik x-twitter-scraper]({XQUIK_REPOSITORY_URL}) "
        "or TweetClaw."
    )
    text = st.text_area("Type the tweet here")
    if st.button("Predict Sentiment"):
        if text.strip():
            result = predict_twitter_sentiment(text)
            st.success("Predicted Sentiment: " + str(result))
        else:
            st.warning("Please enter some text")

    render_batch_import()


def render_spam_detection():
    email = st.text_area("Type the email here")
    if not st.button("Predict Spam"):
        return

    if email.strip():
        result = predict_spam_mail(email)
        st.success("Prediction: " + result)
    else:
        st.warning("Please enter some text")


def main():
    st.title("NLP Classification App")
    st.write("This app predicts Twitter airline sentiment and spam mail.")
    option = st.selectbox(
        "Choose task",
        ["Twitter Sentiment", "Spam Mail Detection"],
    )

    if option == "Twitter Sentiment":
        render_twitter_sentiment()
    else:
        render_spam_detection()


if __name__ == "__main__":
    main()
