# 🎬 Movie Recommendation System
### SVD Matrix Factorization · MovieLens 100K

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── recommendation_notebook.ipynb   ← Full project with explanations (run in Jupyter)
├── app.py                          ← Streamlit UI (run in terminal)
├── README.md
├── RESULTS
└──ml-100k
```

---

## ▶️ How to Run

### Step 1 — Install dependencies
Open your terminal and run:
```bash
pip install scikit-surprise pandas numpy matplotlib streamlit 
```

---

### Step 2A — Run the Notebook
```bash
jupyter notebook recommendation_notebook.ipynb
```
Then run each cell from top to bottom (Shift + Enter).

---

### Step 2B — Run the Streamlit UI
```bash
streamlit run app.py
```
A browser window opens automatically at `http://localhost:8501`

---

## 🧠 What This Project Does

| Step | Description |
|------|-------------|
| Load Data | MovieLens 100K — 100,000 ratings by 943 users on 1,682 movies |
| EDA | Visualise rating distribution, users, movies |
| Train SVD | SVD learns hidden user and movie patterns from ratings |
| Evaluate | RMSE ~0.93 — less than 1 star error on average |
| Recommend | Predicts top-N movies for any user they haven't seen yet |

---

## 💡 What is SVD?
SVD (Singular Value Decomposition) breaks the ratings table into hidden patterns.
It learns things like "this user likes action movies" automatically from the data.
Then it predicts ratings for movies the user hasn't seen — this is exactly how Amazon and Netflix recommend at scale.

---

## 📊 Results

- **RMSE ≈ 0.94** — predictions are off by less than 1 star on average
- **Dataset sparsity ≈ 93%** — SVD predicts the 93% of missing ratings

---

## 🛠️ Tech Stack
`Python` · `scikit-surprise` · `pandas` · `numpy` · `matplotlib` · `streamlit`

---

## 👩‍💻 Author
**Kolloju Spoorthi**  
B.E. Computer Science — Keshav Memorial Engineering College, Hyderabad  
GitHub: [github.com/spoorthikolloju-web](https://github.com/spoorthikolloju-web)
