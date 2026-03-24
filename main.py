import os
import joblib
import re
from model.preprocessing import clean_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
import time
import streamlit as st

# Load model
@st.cache_resource
def model_resources():
    decoder = joblib.load(os.getcwd() + "/model/decoder.joblib")
    model = joblib.load(os.getcwd() + "/model/NB_model.joblib")
    return model, decoder

model, decoder = model_resources()

def infrence(text, model, decoder):
        text = model.predict([text])
        decodded = decoder.inverse_transform(text)
        return decodded[0]


st.title("Movie Review Sentiment Analysis")

    
with st.form(key='review_form'):
    user_review = st.text_area("Enter your Review:", key='review')
    submit_button = st.form_submit_button("Send")

if submit_button:
    with st.spinner("Wait for it...", show_time=True):
        time.sleep(2)
    st.write(infrence(user_review, model, decoder))




