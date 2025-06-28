import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Email Phishing Detector")
st.title("📨 Email Phishing Detector")

email_input = st.text_area("Enter the email content below:")

if st.button("Check Now"):
    if email_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        cleaned = clean_text(email_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 This email is likely PHISHING!")
        else:
            st.success("✅ This email seems LEGITIMATE.")
