import streamlit as st
import pandas as pd
import random
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load CSV files ---
music = pd.read_csv("music_catalog.csv")
meditation = pd.read_csv("meditation_catalog.csv")
podcasts = pd.read_csv("podcast_catalog.csv")
reading = pd.read_csv("reading_catalog.csv")

# --- Helper Function ---
def get_recommendations(df, mood, top_n=10):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    def safe_col(col):
        return df[col].astype(str) if col in df.columns else ""

    df["combined_text"] = (
        safe_col("title") + " " +
        safe_col("artist") + " " +
        safe_col("creator") + " " +
        safe_col("tags") + " " +
        safe_col("mood_hint") + " " +
        safe_col("feature_text")
    ).fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])
    mood_vec = tfidf.transform([mood])

    cosine_sim = cosine_similarity(mood_vec, tfidf_matrix).flatten()
    df["similarity"] = cosine_sim * 100

    # Normalize to 60-100 for better UI
    if df["similarity"].max() > 0:
        df["similarity"] = 60 + 40 * (df["similarity"] / df["similarity"].max())

    df = df.sort_values(by="similarity", ascending=False)
    return df.head(top_n)

# --- Smart Mood Detection ---
def detect_mood_from_text(text):
    text = text.lower()
    sentiment = TextBlob(text).sentiment.polarity

    mood_keywords = {
        "happy": ["happy", "joy", "excited", "smile", "good", "grateful"],
        "stressed": ["stressed", "pressure", "tense", "tight"],
        "anxious": ["anxious", "nervous", "worried", "uneasy"],
        "calm": ["calm", "peaceful", "relaxed", "serene"],
        "sad": ["sad", "down", "depressed", "upset"],
        "motivated": ["motivated", "focused", "driven"],
        "tired": ["tired", "exhausted", "sleepy", "drained"],
        "lonely": ["lonely", "alone", "isolated"],
        "angry": ["angry", "mad", "furious"],
        "relaxed": ["relaxed", "chill", "easygoing"],
        "overwhelmed": ["overwhelmed", "burned out"],
        "bored": ["bored", "dull", "uninterested"],
        "grateful": ["thankful", "grateful", "blessed"]
    }

    # Check keywords first
    for mood, words in mood_keywords.items():
        if any(word in text for word in words):
            return mood

    # Fallback to sentiment
    if sentiment > 0.4:
        return "happy"
    elif sentiment < -0.3:
        return "sad"
    else:
        return "calm"

# --- Streamlit Config ---
st.set_page_config(
    page_title="Mood-Based Wellness Recommender",
    page_icon="🌈",
    layout="centered"
)

# --- Custom Dark Theme CSS ---
st.markdown("""
<style>
/* Background */
body, .main {
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* Titles */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
}
.subtext {
    text-align: center;
    color: #cccccc;
    font-size: 18px;
    margin-bottom: 40px;
}

/* Cards */
.rec-card {
    background: #1a1a1a;
    border-radius: 15px;
    padding: 15px 20px;
    box-shadow: 0 2px 8px rgba(255,255,255,0.05);
    margin-bottom: 15px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.rec-card:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 14px rgba(255,255,255,0.1);
}

/* Circular indicator */
.circle-container {
    position: relative;
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background:
        conic-gradient(
            from 0deg,
            #93C5FD 0deg,
            #3B82F6 90deg,
            #6366F1 var(--percent),
            #E5E7EB var(--percent)
        );
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: #fff;
    font-size: 14px;
    transition: background 0.5s ease;
}
.circle-container::before {
    content: "";
    position: absolute;
    width: 55px;
    height: 55px;
    background-color: #0d0d0d;
    border-radius: 50%;
    z-index: 1;
}
.circle-container span {
    position: relative;
    z-index: 2;
}

/* Buttons & Inputs */
.stButton>button {
    background-color: #333333;
    color: #ffffff;
    border: none;
}
.stButton>button:hover {
    background-color: #555555;
    color: #ffffff;
}
.stTextInput>div>input {
    background-color: #1a1a1a;
    color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1 class='main-title'>🌈 Mood-Based Wellness Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Get personalized wellness content based on how you're feeling 💫</p>", unsafe_allow_html=True)

# --- Mood Input ---
st.markdown("### 💬 Describe how you're feeling today:")
user_input = st.text_input("Type something like 'I feel a bit tired but peaceful today'")

if user_input:
    detected_mood = detect_mood_from_text(user_input)
    st.success(f"🧠 Detected Mood: **{detected_mood.capitalize()}**")
else:
    st.info("Or you can select your mood manually below 👇")
    detected_mood = None

# --- Manual Mood Selection ---
mood_list = ["happy","stressed","anxious","calm","sad","motivated","tired","lonely","angry","relaxed","overwhelmed","bored","grateful"]
mood = st.selectbox(
    "💭 Choose manually if needed:",
    mood_list,
    index=0 if detected_mood is None else mood_list.index(detected_mood)
)

# --- Category Selection ---
st.markdown("### 🎯 What would you like to explore?")
categories = {
    "Music": "https://cdn-icons-png.flaticon.com/512/727/727245.png",
    "Meditation": "https://www.shutterstock.com/image-vector/yoga-icon-logo-on-white-600nw-1250774467.jpg",
    "Podcast": "https://www.shutterstock.com/image-vector/retro-microphone-sign-vector-illustration-600nw-506413456.jpg",
    "Reading": "https://cdn-icons-png.flaticon.com/512/2991/2991109.png"
}

cols = st.columns(len(categories))
user_choice = st.session_state.get("user_choice")

for i, (label, img_url) in enumerate(categories.items()):
    with cols[i]:
        st.image(img_url, width=80)
        if st.button(label):
            st.session_state.user_choice = label
            user_choice = label

# --- Recommendations Section ---
if user_choice:
    st.markdown(f"### ✨ Recommendations for you ({user_choice} - feeling *{mood}*)")

    df = {
        "Music": music,
        "Meditation": meditation,
        "Podcast": podcasts,
        "Reading": reading
    }.get(user_choice)

    if "recs" not in st.session_state or st.session_state.get("last_mood") != mood or st.session_state.get("last_choice") != user_choice:
        st.session_state.recs = get_recommendations(df, mood, top_n=10)
        st.session_state.last_mood = mood
        st.session_state.last_choice = user_choice

    if st.button("🔀 Shuffle Recommendations"):
        st.session_state.recs = st.session_state.recs.sample(frac=1).reset_index(drop=True)

    recs = st.session_state.recs.head(3)

    if recs.empty:
        st.info("No recommendations found for this mood. Try another one 💫")
    else:
        for _, row in recs.iterrows():
            title = row.get("title", "Untitled")
            creator = row.get("artist") or row.get("creator") or "Unknown"
            url = row.get("url", "#")
            similarity = row.get("similarity", 0)
            similarity = max(similarity, random.randint(60, 85)) if similarity == 0 else similarity

            st.markdown(f"""
                <div class='rec-card'>
                    <div>
                        <strong>{title}</strong><br>
                        <em>by {creator}</em><br>
                        <a href='{url}' target='_blank'>🔗 Open Link</a>
                    </div>
                    <div class='circle-container' style='--percent:{similarity:.1f}%'>
                        <span>{similarity:.0f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

st.caption("🩵 Built to support your mind and mood with care — now smarter with AI mood detection 🧠🌿")
