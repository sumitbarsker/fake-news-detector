import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("Fake News Detection System")

st.write("Enter a news article to check whether it is REAL or FAKE")

# Input box
news = st.text_area("Enter News Text")

# Button
if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter some news text")

    else:
        # Convert text
        news_vector = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vector)

        # Show result
        if prediction[0] == "FAKE":
            st.error("This News is FAKE")

        else:
            st.success("This News is REAL")
