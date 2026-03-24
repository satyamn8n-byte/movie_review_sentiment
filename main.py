"""
Simple example: Install packages from requirements.txt
"""
 
import subprocess
import sys
 
def install_requirements(requirements_file='requirements.txt'):
    """Quick way to install packages from requirements.txt"""
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
            check=True
        )
        print("✓ Installation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"✗ requirements.txt not found")
        sys.exit(1)
 
if __name__ == '__main__':
    install_requirements()


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




