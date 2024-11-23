# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 02:32:22 2024

@author: eshra
"""
import streamlit as st
import joblib

# Load the trained model and vectorizer
clf = joblib.load("C:/Users/eshra/OneDrive/Desktop/fake news/fake_news_model.pkl")  # Path to your model file
vectorizer = joblib.load("C:/Users/eshra/OneDrive/Desktop/fake news/tfidf_vectorizer.pkl")  # Path to your vectorizer file

# Function to predict news
def predict_news(news):
    # Preprocess the input news (using the loaded vectorizer)
    processed_news = vectorizer.transform([news])
    
    # Predict using the loaded model
    prediction = clf.predict(processed_news)[0]
    
    # Return result based on the prediction
    return "Real News" if prediction == 0 else "Fake News"

# Streamlit app layout
st.title("Fake News Classifier")
st.write("Enter a news headline to check if it's real or fake!")

# Input text box
news_input = st.text_area("News Headline:")

# Button to trigger prediction
if st.button("Predict"):
    if news_input:
        result = predict_news(news_input)
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a news headline.")

