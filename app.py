import streamlit as st
import pickle
import pandas as pd


with open("spam_nb_model.sav", "rb") as f:
    model = pickle.load(f)

st.title("📩 SMS Spam Detection App (Naïve Bayes)")

st.write("Enter SMS features to predict whether it's **Spam (1)** or **Not Spam (0)**.")


word_freq_free = st.number_input("Word Frequency: 'free'", min_value=0, step=1)
word_freq_win  = st.number_input("Word Frequency: 'win'", min_value=0, step=1)
word_freq_offer = st.number_input("Word Frequency: 'offer'", min_value=0, step=1)
sms_length = st.number_input("SMS Length", min_value=0, step=1)

if st.button("Predict"):
    
    sample = pd.DataFrame(
        [[word_freq_free, word_freq_win, word_freq_offer, sms_length]],
        columns=["word_freq_free", "word_freq_win", "word_freq_offer", "sms_length"]
    )
    
   
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]

    
    if prediction == 1:
        st.error(f"🚨 Prediction: **Spam** (Probability {prob[1]:.2f})")
    else:
        st.success(f"✅ Prediction: **Not Spam** (Probability {prob[0]:.2f})")
