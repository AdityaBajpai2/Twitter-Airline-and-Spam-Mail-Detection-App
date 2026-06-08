import pickle
import streamlit as st

twitter_model = pickle.load(open("twitter_sentiment_pred.sav", "rb"))
twitter_vectorizer = pickle.load(open("twitter_vectorizer.sav", "rb"))
twitter_encoder = pickle.load(open("twitter_label_encoder.sav", "rb"))

fake_model = pickle.load(open("fake_news_analysis.sav", "rb"))
fake_vectorizer = pickle.load(open("fake_vectorizer.sav", "rb"))

spam_model = pickle.load(open("spam_mail_predict.sav", "rb"))
spam_vectorizer = pickle.load(open("spam_vectorizer.sav", "rb"))

def predict_twitter_sentiment(new_tweet):
    X_new = twitter_vectorizer.transform([new_tweet])
    prediction = twitter_model.predict(X_new)
    sentiment = twitter_encoder.inverse_transform(prediction)
    return sentiment[0]

def predict_spam_mail(new_email):
    X_new = spam_vectorizer.transform([new_email])
    prediction = spam_model.predict(X_new)
    if prediction[0] == 0:
        return "Not Spam"
    else:
        return "Spam"

def fake_news_prediction(news):
    X_new = fake_vectorizer.transform([news])
    prediction = fake_model.predict(X_new)
    if prediction[0] == 0:
        return "Real News"
    else:
        return "Fake News"

def main():
    st.title("NLP Classification App")
    st.write("This app predicts tweet sentiment, spam mail, and fake news.")

    option = st.selectbox(
        "Choose task",
        ["Twitter Sentiment", "Spam Mail Detection", "Fake News Detection"]
    )

    if option == "Twitter Sentiment":
        text = st.text_area("Type the tweet here")
        if st.button("Predict Sentiment"):
            if text.strip() == "":
                st.warning("Please enter some text")
            else:
                result = predict_twitter_sentiment(text)
                st.success("Predicted Sentiment: " + result)

    elif option == "Spam Mail Detection":
        email = st.text_area("Type the email here")
        if st.button("Predict Spam"):
            if email.strip() == "":
                st.warning("Please enter some text")
            else:
                result = predict_spam_mail(email)
                st.success("Prediction: " + result)

    elif option == "Fake News Detection":
        news = st.text_area("Type the news text here")
        if st.button("Predict News"):
            if news.strip() == "":
                st.warning("Please enter some text")
            else:
                result = fake_news_prediction(news)
                st.success("Prediction: " + result)

main()