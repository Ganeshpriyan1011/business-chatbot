import os
import json
import ssl
import random
import streamlit as st
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fix SSL issues for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents file
file_path = "updated_green_intents.json"
intents = []
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if "intents" in data:
            intents = data["intents"]
except Exception as e:
    st.error(f"Error loading JSON file: {e}")

# Preprocess training data
patterns, responses, tags = [], [], []
if intents:
    for intent in intents:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            responses.append(random.choice(intent["responses"]))
            tags.append(intent["tag"])

# Train TF-IDF Model
vectorizer = TfidfVectorizer()
if patterns:
    x_train = vectorizer.fit_transform(patterns)
else:
    x_train = None

# Function to log chat
def log_chat(user_input, bot_response):
    log_file = "chat_log.csv"
    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User Input": user_input,
        "Chatbot Response": bot_response
    }
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(log_file, index=False)

# Chatbot Response function
def chatbot(input_text):
    if x_train is None:
        return "Sorry, I cannot process right now."
    input_vec = vectorizer.transform([input_text])
    similarity_scores = cosine_similarity(input_vec, x_train).flatten()
    best_match_index = np.argmax(similarity_scores)
    confidence = similarity_scores[best_match_index]
    if confidence < 0.3:
        return "I'm sorry, I don't understand. Can you rephrase?"
    return responses[best_match_index]

# Business Assessment Functions
def run_assessment():
    responses = {}

    questions = [
        ("business_name", "What is the name of your business?"),
        ("industry", "What industry does your business operate in?"),
        ("employees", "How many employees work in your company? (Enter a number)"),
        ("energy_source", "What is your primary energy source? (Renewable, Fossil fuels, Mixed)"),
        ("waste_recycling", "Do you use a waste recycling system? (Yes/No)"),
        ("recycling_percentage", "What percentage of waste is recycled? (e.g., 20, 50, 80)"),
        ("water_conservation", "Do you implement water conservation measures? (Yes/No)"),
        ("carbon_tracking", "Do you track your carbon emissions? (Yes/No)"),
        ("green_certifications", "Do you hold any green certifications? (Yes/No)"),
        ("sustainability_goals", "Are you working towards any sustainability goals? (e.g., Net Zero, Carbon Neutral)")
    ]

    for key, prompt in questions:
        responses[key] = st.text_input(prompt, key=key)

    if st.button("Submit Assessment"):
        score = calculate_score(responses)
        category = categorize_business(score)
        st.success(f"Assessment Complete! Your sustainability score is **{score}**.")
        st.info(f"Your business is categorized as: **{category}**.")

def calculate_score(responses):
    score = 0
    if responses.get("energy_source", "").lower() == "renewable":
        score += 20
    if responses.get("waste_recycling", "").lower() == "yes":
        score += 10
    if responses.get("water_conservation", "").lower() == "yes":
        score += 10
    if responses.get("carbon_tracking", "").lower() == "yes":
        score += 10
    if responses.get("green_certifications", "").lower() == "yes":
        score += 20
    return score

def categorize_business(score):
    if score >= 50:
        return "Sustainable Business"
    elif score >= 30:
        return "Moderately Sustainable"
    else:
        return "Needs Improvement"

# Main App
def main():
    st.title("🌱 Green Business Consultant Chatbot")

    menu = ["Chat", "Assessment", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if choice == "Chat":
        st.subheader("Start Chatting")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["text"])

        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            bot_response = chatbot(user_input)
            st.session_state.chat_history.append({"role": "assistant", "text": bot_response})
            with st.chat_message("assistant"):
                st.write(bot_response)

            log_chat(user_input, bot_response)

    elif choice == "Assessment":
        st.subheader("Green Business Assessment")
        run_assessment()

    elif choice == "Conversation History":
        st.subheader("Conversation History")
        for msg in st.session_state.chat_history:
            st.text(f"{msg['role'].title()}: {msg['text']}")

    elif choice == "About":
        st.subheader("About this Chatbot")
        st.write("""
        This chatbot helps businesses assess and improve their sustainability practices.
        Features:
        - Smart chat answering about green practices 🌱
        - Quick business sustainability assessment 📋
        - Tips and advice on making your business eco-friendly 🌍
        """)

if __name__ == "__main__":
    main()
