# 🎬 CineMatch — ML Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=flat-square)
## 🚀 [Live Demo →](https://cinematch-wteg9vhafflamkyw2ubkka.streamlit.app/)

> A content-based movie recommendation engine built with **TF-IDF vectorization** and **Cosine Similarity** — the same NLP technique used by Netflix, Spotify, and Google News.

---

## 🧠 How It Works

This project implements a **content-based filtering** recommendation system from scratch using real NLP:

```
Movie Description + Genre
        ↓
  TfidfVectorizer          ← converts text into numerical vectors
  (unigrams + bigrams)       weights rare words higher than common ones
        ↓
  TF-IDF Matrix            ← shape: (n_movies × n_vocab_features)
        ↓
  Cosine Similarity        ← measures angle between two movie vectors
        ↓
  Top-N Recommendations    ← movies with smallest angular distance
```

### Key Concepts

| Concept | What it does |
|---|---|
| **TF (Term Frequency)** | How often a word appears in this movie's text |
| **IDF (Inverse Document Frequency)** | Penalises words common to ALL movies (e.g. "the", "a") |
| **TF-IDF Score** | TF × IDF — high score = word is important to THIS movie specifically |
| **Cosine Similarity** | Measures how similar two TF-IDF vectors are (1 = identical, 0 = unrelated) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
cinematch/
│
├── app.py              # Main Streamlit app (UI + ML pipeline)
├── requirements.txt    # Python dependencies
└── README.md           # You are here
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** — `TfidfVectorizer`, `cosine_similarity`
- **Pandas** — dataset management
- **Streamlit** — interactive web UI

---

## 📊 Dataset

40 hand-curated movies across genres: Sci-Fi, Drama, Thriller, Horror, Comedy, Animation, and more.

Each movie has:
- `title` — movie name
- `year` — release year
- `genre` — primary genre(s)
- `desc` — plot description used for vectorization

---

## 💡 What I Learned

- How **TF-IDF** converts raw text into meaningful numerical representations
- Why **cosine similarity** works better than Euclidean distance for text vectors
- How to build a real **NLP pipeline** using Scikit-learn
- How to deploy an interactive ML app with **Streamlit**

---

## 📌 Next Steps

- [ ] Add a larger dataset (TMDB / MovieLens)
- [ ] Add collaborative filtering (user-based recommendations)
- [ ] Deploy on Streamlit Community Cloud
- [ ] Add poster images via TMDB API

---

## 🙋 Author

Built by **[Your Name]** as part of a self-directed ML learning journey.

Connect with me on [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)

---

*If you found this useful, give it a ⭐ — it helps others find it!*
