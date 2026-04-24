import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch – ML Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    h1 { font-size: 2.4rem !important; }
    .stTextInput > div > div > input {
        background-color: #12121a;
        color: #f0ede6;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .movie-card {
        background: #1a1a26;
        border: 1px solid #2a2a3a;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 14px;
    }
    .rank { color: rgba(232,168,56,0.35); font-size: 2rem; font-weight: 700; }
    .movie-title { font-size: 1.2rem; font-weight: 600; color: #f0ede6; }
    .genre-tag {
        background: rgba(232,168,56,0.12);
        color: #e8a838;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .score { color: #e8a838; font-weight: 600; font-size: 0.9rem; }
    .desc { color: #8a8590; font-size: 0.88rem; line-height: 1.6; }
    .pill {
        display: inline-block;
        background: rgba(232,168,56,0.12);
        color: #e8a838;
        border: 1px solid rgba(232,168,56,0.2);
        border-radius: 100px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-right: 6px;
    }
    .info-box {
        background: #12121a;
        border-left: 3px solid #e8a838;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #8a8590;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)


# ── Dataset ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies = [
        {"title": "Inception",                        "year": 2010, "genre": "Sci-Fi Thriller",      "desc": "A thief steals corporate secrets through dream-sharing technology and is given the task of planting an idea. Reality dreams layers subconscious visual spectacle mind-bending."},
        {"title": "Interstellar",                     "year": 2014, "genre": "Sci-Fi Drama",          "desc": "Explorers travel through a wormhole in space to ensure humanity's survival. Time dilation relativity black holes gravity wormhole space survival love."},
        {"title": "The Dark Knight",                  "year": 2008, "genre": "Action Superhero",      "desc": "Batman faces the Joker, a criminal mastermind plunging Gotham into chaos. Moral dilemmas heroism justice crime gritty comic book villain."},
        {"title": "Parasite",                         "year": 2019, "genre": "Thriller Drama",        "desc": "A poor family infiltrates a wealthy household through deception. Class inequality wealth poverty dark comedy social satire Korean cinema."},
        {"title": "The Shawshank Redemption",         "year": 1994, "genre": "Drama",                 "desc": "Two imprisoned men bond over years finding hope and redemption. Friendship prison injustice perseverance freedom hope kindness."},
        {"title": "Pulp Fiction",                     "year": 1994, "genre": "Crime Thriller",        "desc": "Criminal lives in Los Angeles intertwine across four tales of violence and redemption. Non-linear storytelling dialogue crime murder dark comedy."},
        {"title": "The Matrix",                       "year": 1999, "genre": "Sci-Fi Action",         "desc": "A hacker discovers reality is a simulation and joins a rebellion against machine overlords. Virtual reality dystopia AI philosophy action martial arts."},
        {"title": "Avengers: Endgame",                "year": 2019, "genre": "Action Superhero",      "desc": "The Avengers assemble to reverse Thanos actions and restore balance to the universe. Superheroes teamwork time travel sacrifice action spectacle comic."},
        {"title": "Spirited Away",                    "year": 2001, "genre": "Fantasy Animation",     "desc": "A young girl enters a spirit world while her parents are transformed. Japanese folklore spirits magic animation coming of age adventure."},
        {"title": "Get Out",                          "year": 2017, "genre": "Horror Thriller",       "desc": "A Black man visits his white girlfriend's family and uncovers disturbing secrets. Racial horror social commentary suspense psychological thriller."},
        {"title": "Mad Max: Fury Road",               "year": 2015, "genre": "Action Adventure",      "desc": "In a post-apocalyptic wasteland a woman rebels against a tyrant while racing across a desert. Survival vehicles explosion chase pursuit action."},
        {"title": "The Godfather",                    "year": 1972, "genre": "Crime Drama",           "desc": "A crime dynasty transfers control to its reluctant son. Mafia family loyalty power crime drama organized crime Italian-American."},
        {"title": "Blade Runner 2049",                "year": 2017, "genre": "Sci-Fi Drama",          "desc": "A blade runner discovers a long-buried secret threatening society. Dystopia androids AI memory identity future replicants."},
        {"title": "Hereditary",                       "year": 2018, "genre": "Horror Drama",          "desc": "A family unravels following the death of their secretive grandmother. Occult grief trauma family supernatural horror terrifying cult."},
        {"title": "Whiplash",                         "year": 2014, "genre": "Drama Music",           "desc": "A promising drummer encounters a ruthless instructor at a music conservatory. Ambition obsession talent mentor music jazz competition."},
        {"title": "La La Land",                       "year": 2016, "genre": "Romance Musical",       "desc": "A musician and actress fall in love while chasing their dreams in Los Angeles. Jazz romance dreams ambition musical dancing nostalgia."},
        {"title": "Knives Out",                       "year": 2019, "genre": "Mystery Thriller",      "desc": "A detective investigates the death of a wealthy crime novelist. Whodunit mystery suspects family money inheritance detective comedy."},
        {"title": "Arrival",                          "year": 2016, "genre": "Sci-Fi Drama",          "desc": "A linguist communicates with extraterrestrial visitors. Aliens language time consciousness communication humanity first contact."},
        {"title": "The Silence of the Lambs",         "year": 1991, "genre": "Crime Thriller",        "desc": "An FBI trainee seeks help from imprisoned Hannibal Lecter to catch a serial killer. Serial killer psychological manipulation suspense crime FBI."},
        {"title": "Fight Club",                       "year": 1999, "genre": "Thriller Drama",        "desc": "An insomniac and a soap salesman form an underground fighting club that evolves into something dangerous. Identity masculinity anarchism twist."},
        {"title": "Eternal Sunshine of the Spotless Mind", "year": 2004, "genre": "Romance Sci-Fi",   "desc": "A couple undergo a procedure to erase each other from their memories. Memory love loss relationships identity nostalgia erasing the past."},
        {"title": "Everything Everywhere All at Once","year": 2022, "genre": "Sci-Fi Comedy",         "desc": "A middle-aged Chinese-American woman explores parallel universes to save existence. Multiverse absurdist comedy existential family immigration."},
        {"title": "Oppenheimer",                      "year": 2023, "genre": "Historical Drama",      "desc": "The story of J. Robert Oppenheimer and the creation of the atomic bomb. Science morality war nuclear physics biography history destruction."},
        {"title": "1917",                             "year": 2019, "genre": "War Drama",             "desc": "During World War One two soldiers cross enemy territory to deliver a life-saving message. War survival real-time tension soldiers mission courage."},
        {"title": "Dune",                             "year": 2021, "genre": "Sci-Fi Epic",           "desc": "A noble family is embroiled in a war for control of a desert planet with valuable spice. Epic fantasy sci-fi politics religion destiny sandworm."},
        {"title": "Joker",                            "year": 2019, "genre": "Drama Thriller",        "desc": "Failed comedian Arthur Fleck descends into madness becoming the Joker. Mental illness society origin villain crime Gotham tragedy dark."},
        {"title": "Coco",                             "year": 2017, "genre": "Animation Family",      "desc": "A young boy enters the land of the dead to connect with his ancestors and pursue music. Family heritage death Mexico animation musical heartwarming."},
        {"title": "Tenet",                            "year": 2020, "genre": "Sci-Fi Action",         "desc": "A secret agent uses time inversion to prevent World War III. Time manipulation espionage action paradox physics secret agent."},
        {"title": "Midsommar",                        "year": 2019, "genre": "Horror Drama",          "desc": "A couple travels to Sweden for a festival that turns sinister. Folk horror cult ritual summer daylight grief relationship breakup."},
        {"title": "The Grand Budapest Hotel",         "year": 2014, "genre": "Comedy Drama",          "desc": "A legendary hotel concierge teams with a lobby boy to prove his innocence in a murder. Quirky comedy mystery art style Europe hotel."},
        {"title": "12 Angry Men",                     "year": 1957, "genre": "Drama Thriller",        "desc": "A lone juror challenges others to review evidence in a murder trial. Justice law persuasion bias group dynamics dialogue tension courtroom."},
        {"title": "No Country for Old Men",           "year": 2007, "genre": "Crime Thriller",        "desc": "Violence and fate engulf a rancher who finds drug money. Dark existential crime relentless villain fate morality desert."},
        {"title": "Pan's Labyrinth",                  "year": 2006, "genre": "Fantasy Drama",         "desc": "In post-war Spain a young girl escapes reality through a dark fairy tale labyrinth. Fantasy dark fairy tale war imagination Spain mythical creature."},
        {"title": "Her",                              "year": 2013, "genre": "Romance Sci-Fi",        "desc": "A writer develops a romantic relationship with an AI operating system. Artificial intelligence loneliness love technology future consciousness."},
        {"title": "Spider-Man: Into the Spider-Verse","year": 2018, "genre": "Animation Superhero",   "desc": "A teenager gains spider powers and meets Spider-People from parallel universes. Multiverse superhero animation comic identity hero youth."},
        {"title": "The Revenant",                     "year": 2015, "genre": "Adventure Drama",       "desc": "A frontiersman fights for survival after being mauled by a bear. Survival nature revenge wilderness endurance frontier."},
        {"title": "A Beautiful Mind",                 "year": 2001, "genre": "Drama Biography",       "desc": "The story of Nobel Prize winner John Nash and his struggles with mental illness. Mathematics genius schizophrenia biography love perseverance."},
        {"title": "The Prestige",                     "year": 2006, "genre": "Mystery Thriller",      "desc": "Two rival magicians battle to create the ultimate illusion. Magic obsession rivalry deception mystery twist science illusion."},
        {"title": "Gone Girl",                        "year": 2014, "genre": "Thriller Drama",        "desc": "A man becomes the prime suspect when his wife mysteriously disappears. Marriage deception media manipulation mystery suspense dark twist."},
        {"title": "Black Swan",                       "year": 2010, "genre": "Thriller Drama",        "desc": "A ballet dancer falls into obsession while preparing for Swan Lake. Perfectionism obsession identity duality psychological thriller dance."},
    ]
    return pd.DataFrame(movies)


# ── ML Model ──────────────────────────────────────────────────
@st.cache_resource
def build_model(df):
    """
    Build TF-IDF matrix and return vectorizer + matrix.

    Pipeline:
    1. Combine genre (weighted 2x) + description into one text field
    2. Fit TfidfVectorizer (removes stop words, uses unigrams + bigrams)
    3. Transform all movie texts into TF-IDF vectors
    4. At query time: compute cosine_similarity between query vector and all rows
    """
    df["corpus"] = df["genre"] + " " + df["genre"] + " " + df["desc"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=5000,
        sublinear_tf=True,    # apply log normalization to TF
    )
    tfidf_matrix = vectorizer.fit_transform(df["corpus"])
    return vectorizer, tfidf_matrix


def get_recommendations(title, df, vectorizer, tfidf_matrix, n=5):
    """Return top-n similar movies to `title` using cosine similarity."""
    idx = df[df["title"].str.lower() == title.lower()].index
    if len(idx) == 0:
        return None, None

    idx = idx[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0  # exclude the movie itself

    top_indices = sim_scores.argsort()[::-1][:n]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]
    results["match_pct"] = (results["similarity"] / results["similarity"].max() * 100).round(0).astype(int)
    return results, df.iloc[idx]


# ── UI ────────────────────────────────────────────────────────
df = load_data()
vectorizer, tfidf_matrix = build_model(df)

# Header
st.markdown("## 🎬 CineMatch")
st.markdown("### Find your next *favourite film*")
st.markdown(
    '<span class="pill">TF-IDF</span>'
    '<span class="pill">Cosine Similarity</span>'
    '<span class="pill">NLP</span>'
    '<span class="pill">Scikit-learn</span>',
    unsafe_allow_html=True
)
st.markdown("---")

st.markdown(
    '<div class="info-box"><strong>How it works:</strong> Each movie\'s genre + description is converted '
    'into a TF-IDF vector using <code>TfidfVectorizer</code> with bigrams. The algorithm computes '
    '<strong>cosine similarity</strong> between your chosen movie and all others — higher similarity = '
    'closer angle between vectors. Same technique used by Netflix, Spotify, and Google News.</div>',
    unsafe_allow_html=True
)
st.markdown("")

# Search
all_titles = sorted(df["title"].tolist())
selected = st.selectbox(
    "🎥 Choose a movie you love",
    options=[""] + all_titles,
    index=0,
    help="Select from 40 curated films"
)

col1, col2 = st.columns([3, 1])
with col2:
    n_results = st.slider("Results", 3, 8, 5)

if selected:
    results, base = get_recommendations(selected, df, vectorizer, tfidf_matrix, n=n_results)

    if results is None:
        st.error("Movie not found. Please select from the dropdown.")
    else:
        st.markdown(f"**Recommendations based on:** *{base['title']}* ({base['year']}) · {base['genre']}")
        st.markdown("---")

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            bar = "█" * int(row["match_pct"] / 10) + "░" * (10 - int(row["match_pct"] / 10))
            st.markdown(f"""
            <div class="movie-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start">
                    <div>
                        <span class="rank">0{rank}</span>
                        <span class="movie-title"> {row['title']}</span>
                        &nbsp;<span class="genre-tag">{row['genre']}</span>
                        <span style="color:#8a8590; font-size:0.8rem; margin-left:8px">{row['year']}</span>
                    </div>
                    <div style="text-align:right">
                        <div class="score">{row['match_pct']}% match</div>
                        <div style="color:#e8a838; font-size:0.7rem; letter-spacing:0.05em">{bar}</div>
                    </div>
                </div>
                <div class="desc" style="margin-top:0.6rem">{row['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='color:#8a8590; font-size:0.78rem; text-align:center'>"
    "Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp; "
    "TF-IDF + Cosine Similarity · 40 movies dataset</div>",
    unsafe_allow_html=True
)
