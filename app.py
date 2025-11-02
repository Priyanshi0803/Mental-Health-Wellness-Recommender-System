import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------- Helper function ----------
def safe_read_csv(path):
    if not os.path.exists(path):
        st.warning(f"⚠️ File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

# ---------- Load Data ----------
music_df = safe_read_csv("music_catalog.csv")
med_df = safe_read_csv("meditation_catalog.csv")
pod_df = safe_read_csv("podcast_catalog.csv")
read_df = safe_read_csv("reading_catalog.csv")

# ---------- Combine all for reference ----------
all_data = {
    "Music": music_df,
    "Meditation": med_df,
    "Podcast": pod_df,
    "Reading": read_df
}

# ---------- TF-IDF setup ----------
def prepare_vectorizer(df):
    if df.empty:
        return None, None
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(df['feature_text'].fillna(''))
    return vectorizer, tfidf

# ---------- Recommend based on mood ----------
def recommend(df, vectorizer, tfidf_matrix, user_mood):
    if df.empty or vectorizer is None:
        return pd.DataFrame()
    user_vec = vectorizer.transform([user_mood])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    indices = similarity.argsort()[0][-5:][::-1]
    return df.iloc[indices][["type", "title", "creator", "url"]]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🧘‍♀️ Wellness Recommender", layout="centered")

st.title("🧠 Mental Health & Wellness Recommender")
st.markdown("### Get personalized **music, meditation, podcasts, or reading** suggestions 🎶🧘‍♀️🎙️📖")

# Step 1: Ask what user wants
content_type = st.selectbox(
    "👉 What type of content do you want?",
    ["Music", "Meditation", "Podcast", "Reading"]
)

# Step 2: Ask mood
mood = st.selectbox(
    "💬 How are you feeling right now?",
    [
        "happy", "stressed", "anxious", "calm", "sad",
        "motivated", "tired", "lonely", "angry",
        "relaxed", "overwhelmed", "bored", "grateful"
    ]
)


# Step 3: Button to get recommendations
if st.button("🎧 Show Recommendations"):
    df = all_data.get(content_type)
    if df is not None and not df.empty:
        vectorizer, tfidf_matrix = prepare_vectorizer(df)
        results = recommend(df, vectorizer, tfidf_matrix, mood)

        if not results.empty:
            st.subheader(f"✨ Recommended {content_type}s for '{mood}' mood:")
            for _, row in results.iterrows():
                st.markdown(f"""
                **📝 Title:** {row['title']}  
                **👤 Creator:** {row['creator']}  
                👉 [Open Link]({row['url']})  
                ---
                """)
        else:
            st.error("No matching recommendations found.")
    else:
        st.warning(f"No data available for {content_type} 😞")

st.caption("💚 Built with Streamlit — personalized wellness recommendations powered by AI.")
