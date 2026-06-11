

import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
import urllib.request
import zipfile
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #f0f0f0; }
    h1 { color: #e2b96f !important; text-align: center; letter-spacing: 2px; }
    h3 { color: #e2b96f !important; }
    [data-testid="metric-container"] {
        background: #1c1c2e; border: 1px solid #2e2e4a;
        border-radius: 10px; padding: 12px;
    }
    [data-testid="metric-container"] label { color: #aaaacc !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2b96f !important; font-size: 1.6rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e2b96f, #c49a45);
        color: #0f0f1a; font-weight: 700; border: none;
        border-radius: 8px; padding: 10px 28px; width: 100%;
    }
    .movie-card {
        background: #1c1c2e; border: 1px solid #2e2e4a;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .info-box {
        background: #1c1c2e; border-left: 4px solid #e2b96f;
        border-radius: 6px; padding: 12px 16px; margin: 10px 0;
        font-size: 0.88rem; color: #aaaacc; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


#Load & Train (runs only once
@st.cache_resource(show_spinner=False)
def load_and_train():
    print("before")
    data = Dataset.load_builtin('ml-100k')
    print("after")
# Download movie titles file
    if not os.path.exists('ml-100k/u.item'):
     url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
     urllib.request.urlretrieve(url, 'ml-100k.zip')
     with zipfile.ZipFile('ml-100k.zip', 'r') as z:
        z.extract('ml-100k/u.item')

# Load movie names
    movies_df = pd.read_csv(
       'ml-100k/u.item',
        sep='|',
       encoding='latin-1',
       header=None,
       usecols=[0, 1],
       names=['movie_id', 'title']
    )
    movie_names = dict(zip(movies_df['movie_id'], movies_df['title']))
    trainset, testset = train_test_split(data, test_size=0.20, random_state=42)

    model = SVD(n_factors=100, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)

    preds = model.test(testset)
    rmse  = round(accuracy.rmse(preds, verbose=False), 4)
    mae   = round(accuracy.mae(preds,  verbose=False), 4)

    df = pd.DataFrame(data.raw_ratings, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    df['user_id']  = df['user_id'].astype(int)
    df['movie_id'] = df['movie_id'].astype(int)
    df['rating']   = df['rating'].astype(float)

    return model, df, rmse, mae, movie_names


def get_recommendations(model, df, user_id, n, movie_names):
    already_seen  = set(df[df['user_id'] == user_id]['movie_id'].tolist())
    all_movies    = df['movie_id'].unique().tolist()
    unseen        = [m for m in all_movies if m not in already_seen]

    preds = [(mid, model.predict(str(user_id), str(mid)).est) for mid in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]


def to_stars(rating):
    return "★" * int(round(rating)) + "☆" * (5 - int(round(rating)))


# UI
st.markdown("# 🎬 Movie Recommender")
st.markdown("<p style='text-align:center;color:#aaaacc'>SVD Matrix Factorization · MovieLens 100K</p>", unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Training SVD model... (first run only, ~30 sec)"):
    model, df, rmse, mae, movie_names = load_and_train()

st.markdown("### 📊 Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("Total Ratings", "100,000")
c2.metric("RMSE", str(rmse))
c3.metric("MAE",  str(mae))

st.markdown("""
<div class='info-box'>
    <b>RMSE ~0.94</b> means our predicted ratings are off by less than 1 star on average. 
    Lower is better. This is how Amazon recommends products at scale!
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🎯 Get Recommendations")

user_id = st.selectbox("Select User ID", sorted(df['user_id'].unique().tolist()))
# Show user's past ratings
st.markdown("#### 🎭 This User's Taste (recently rated movies)")
user_history = df[df['user_id'] == user_id].sort_values('rating', ascending=False).head(8)

for _, row in user_history.iterrows():
    title = movie_names.get(int(row['movie_id']), 'Unknown')
    stars_str = to_stars(row['rating'])
    st.markdown(f"""
    <div class='movie-card'>
        <span style='flex:1; color:#aaaacc'>{title}</span>
        <span style='color:#e2b96f'>{stars_str} &nbsp; {row['rating']:.1f} / 5.00</span>
    </div>
    """, unsafe_allow_html=True)
n_recs  = st.slider("Number of recommendations", 5, 20, 10)

if st.button("🎬 Recommend Movies"):
    recs    = get_recommendations(model, df, user_id, n_recs, movie_names)
    n_rated = len(df[df['user_id'] == user_id])

    st.markdown(f"""
    <div class='info-box'>
        User <b>{user_id}</b> has rated <b>{n_rated} movies</b>.
        SVD learned their taste and is predicting scores for all unseen movies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"#### Top {n_recs} Movies for User {user_id}")
    for rank, (mid, rating) in enumerate(recs, 1):
        title = movie_names.get(int(mid), 'Unknown')
        st.markdown(f"""
           <div class='movie-card'>
           <span style='color:#aaaacc'>#{rank} &nbsp;</span>
           <span style='flex:1'>{title} <span style='color:#e2b96f'>{to_stars(rating)}</span></span>
           <span style='color:#e2b96f; font-weight:700'>{rating:.2f} / 5.00</span>
           </div>
    """, unsafe_allow_html=True)

st.markdown("---")
with st.expander(" How does SVD work?"):
    st.markdown("""
    SVD breaks the ratings table into hidden patterns.

    - Each **user** gets a hidden vector → their taste profile (e.g. likes action, dislikes romance)
    - Each **movie** gets a hidden vector → its feature profile (e.g. 80% action, 10% romance)
    - SVD **learns** these automatically from ratings — no manual labels needed

    **Prediction = user vector · movie vector + biases**

    This is why it scales to millions of users — Amazon and Netflix use the same idea!
    """)
