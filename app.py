import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data ---
music = pd.read_csv("music_catalog.csv")
meditation = pd.read_csv("meditation_catalog.csv")
podcasts = pd.read_csv("podcast_catalog.csv")
reading = pd.read_csv("reading_catalog.csv")

# --- Helper function for recommendations ---
def get_recommendations(df, mood, top_n=3):
    """Filter content by mood or fallback to similarity search"""
    if 'mood_hint' not in df.columns:
        return df.sample(top_n)
    
    mood_filtered = df[df['mood_hint'].str.lower().str.contains(mood.lower(), na=False)]
    
    if mood_filtered.empty:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['feature_text'].fillna(''))
        mood_vec = tfidf.transform([mood])
        cosine_sim = cosine_similarity(mood_vec, tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[-top_n:][::-1]
        mood_filtered = df.iloc[top_indices]
    
    return mood_filtered.sample(min(top_n, len(mood_filtered)))


# --- Streamlit App ---
st.set_page_config(page_title="Wellness Recommender", page_icon="🧘‍♀️", layout="centered")
st.title("🌈 Mood-Based Wellness Recommender")

# Step 1: Ask mood
mood = st.selectbox(
    "💬 How are you feeling today?",
    [
        "happy", "stressed", "anxious", "calm", "sad",
        "motivated", "tired", "lonely", "angry",
        "relaxed", "overwhelmed", "bored", "grateful"
    ]
)

# Step 2: Ask what the user wants
content_type = st.radio(
    "🎯 What would you like to explore?",
    ["Music 🎵", "Meditation 🧘‍♀️", "Podcast 🎙️", "Reading 📖"]
)

if st.button("✨ Get My Recommendations"):
    st.markdown(f"### You’re feeling *{mood.capitalize()}* and want {content_type} 💖")

    if "Music" in content_type:
        recs = get_recommendations(music, mood)
        st.subheader("🎵 Music Recommendations")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}** by *{row['artist']}*  \n  🔗 [Listen here]({row['url']})")

    elif "Meditation" in content_type:
        recs = get_recommendations(meditation, mood)
        st.subheader("🧘 Guided Meditations")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}**  \n  🔗 [Relax here]({row['url']})")

    elif "Podcast" in content_type:
        recs = get_recommendations(podcasts, mood)
        st.subheader("🎙️ Podcasts")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}** by *{row['host']}*  \n  🔗 [Listen here]({row['url']})")

    elif "Reading" in content_type:
        recs = get_recommendations(reading, mood)
        st.subheader("📖 Reading Material")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}**  \n  ✍️ {row['creator']}  \n  🔗 [Read here]({row['url']})")

st.caption("🩵 Built to support your mind and mood with care.")
