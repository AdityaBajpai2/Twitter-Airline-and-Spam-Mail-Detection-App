import pickle
import streamlit as st

airline_vectorizer = pickle.load(open("airline_vectorizer.pkl", "rb"))
sentiment_model = pickle.load(open("twitter_sentiment_pred.sav", "rb"))

spam_model = pickle.load(open("spam_mail_predict.sav", "rb"))
spam_vectorizer = pickle.load(open("spam_vectorizer.pkl", "rb"))

twitter_encoder = pickle.load(open("twitter_encoder.sav", "rb"))

def predict_twitter_sentiment(new_tweet):
    X_new = airline_vectorizer.transform([new_tweet])
    prediction = sentiment_model.predict(X_new)
    sentiment = twitter_encoder.inverse_transform(prediction)
    return sentiment[0]

def predict_spam_mail(new_email):
    X_new = spam_vectorizer.transform([new_email])
    prediction = spam_model.predict(X_new)
    if prediction[0] == 0:
        return "Not Spam"
    else:
        return "Spam"

def main():
    st.title("NLP Classification App")
    st.write("This app predicts tweet sentiment and spam mail.")

    option = st.selectbox(
        "Choose task",
        ["Twitter Sentiment", "Spam Mail Detection"]
    )

    if option == "Twitter Sentiment":
        text = st.text_area("Type the tweet here")
        if st.button("Predict Sentiment"):
            if text.strip() == "":
                st.warning("Please enter some text")
            else:
                result = predict_twitter_sentiment(text)
                st.success("Predicted Sentiment: " + str(result))

    elif option == "Spam Mail Detection":
        email = st.text_area("Type the email here")
        if st.button("Predict Spam"):
            if email.strip() == "":
                st.warning("Please enter some text")
            else:
                result = predict_spam_mail(email)
                st.success("Prediction: " + result)

if __name__ == "__main__":
    main()
