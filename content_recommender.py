import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load all datasets
music_df = pd.read_csv("music_catalog.csv")
med_df = pd.read_csv("meditation_catalog.csv")
pod_df = pd.read_csv("podcast_catalog.csv")
read_df = pd.read_csv("reading_catalog.csv")

# Combine them
catalog = pd.concat([music_df, med_df, pod_df, read_df], ignore_index=True)

# Create TF-IDF matrix based on 'feature_text'
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(catalog["feature_text"])

def recommend_by_mood(mood_label, top_n=5):
    """Return top-N recommendations for a given mood."""
    subset = catalog[catalog["mood_hint"].str.lower() == mood_label.lower()]
    if subset.empty:
        print("Mood not found. Showing general recommendations.\n")
        subset = catalog

    # Get similarity scores
    subset_vec = tfidf.transform(subset["feature_text"])
    sim = cosine_similarity(subset_vec, tfidf_matrix)
    sim_scores = sim.mean(axis=0)
    catalog["similarity"] = sim_scores

    # Sort and show results
    recs = catalog.sort_values(by="similarity", ascending=False)
    return recs.head(top_n)[["type", "title", "creator", "url", "mood_hint", "similarity"]]

# Example test
if __name__ == "__main__":
    mood = input("Enter your mood (happy, sad, stressed, anxious, calm): ")
    results = recommend_by_mood(mood, top_n=5)
    print("\nTop Recommendations:")
    print(results.to_string(index=False))
