import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load your CSV data ---
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

# --- Streamlit UI Config ---
st.set_page_config(page_title="Mood-Based Wellness Recommender", page_icon="🌈", layout="centered")

# --- Custom CSS for beautiful layout ---
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #222;
    }
    .subtext {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .option-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 20px;
    }
    .option {
        transition: transform 0.2s ease, box-shadow 0.3s ease;
        border-radius: 15px;
        padding: 10px;
        background-color: #fff;
        text-align: center;
        width: 140px;
        cursor: pointer;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    }
    .option:hover {
        transform: scale(1.1);
        box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
    }
    img {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title + Intro Text ---
st.markdown("<h1 class='main-title'>🌈 Mood-Based Wellness Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Get personalized wellness content based on how you're feeling 💫</p>", unsafe_allow_html=True)

# --- Mood Selection ---
mood = st.selectbox(
    "💭 How are you feeling today?",
    [
        "happy", "stressed", "anxious", "calm", "sad",
        "motivated", "tired", "lonely", "angry",
        "relaxed", "overwhelmed", "bored", "grateful"
    ]
)

st.markdown("### 🎯 What would you like to explore?")

# --- Category Icons ---
categories = {
    "Music": "https://cdn-icons-png.flaticon.com/512/727/727245.png",
    "Meditation": "https://cdn-icons-png.flaticon.com/512/3028/3028707.png",
    "Podcast": "https://cdn-icons-png.flaticon.com/512/727/727240.png",
    "Reading": "https://cdn-icons-png.flaticon.com/512/2991/2991109.png"
}

# --- Interactive Category Buttons ---
cols = st.columns(len(categories))
user_choice = None

for i, (label, img_url) in enumerate(categories.items()):
    with cols[i]:
        st.image(img_url, width=80)
        if st.button(label):
            user_choice = label

# --- Display Recommendations ---
if user_choice:
    st.markdown(f"### ✨ Recommendations for you ({user_choice} - feeling {mood})")

    if user_choice == "Music":
        recs = get_recommendations(music, mood)
        st.subheader("🎵 Music Recommendations")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}** by *{row['artist']}*  \n  🔗 [Listen here]({row['url']})")

    elif user_choice == "Meditation":
        recs = get_recommendations(meditation, mood)
        st.subheader("🧘 Guided Meditations")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}**  \n  🔗 [Relax here]({row['url']})")

    elif user_choice == "Podcast":
        recs = get_recommendations(podcasts, mood)
        st.subheader("🎙️ Podcasts")
        for _, row in recs.iterrows():
            host = row['creator'] if 'creator' in row else row.get('host', 'Unknown')
            st.markdown(f"- **{row['title']}** by *{host}*  \n  🔗 [Listen here]({row['url']})")

    elif user_choice == "Reading":
        recs = get_recommendations(reading, mood)
        st.subheader("📖 Reading Material")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}** by *{row['creator']}*  \n  🔗 [Read here]({row['url']})")

    if recs.empty:
        st.info("No specific recommendations found for this mood — try another one! 💫")

st.caption("🩵 Built to support your mind and mood with care.")
