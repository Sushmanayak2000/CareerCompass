# # import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pdfplumber
import re
import nltk
from html import escape as esc

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="CareerCompass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Theme — pastel palette look
# -----------------------------
def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..900;1,9..144,400..900&family=Source+Sans+3:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        :root {
            --baby-blue: #EAF6FF;
            --baby-blue-deep: #D8ECFA;
            --lavender-grey: #D4D5DF;
            --mint: #98D8BF;
            --periwinkle: #8FA1E0;
            --coral: #F2967E;
            --slate: #617086;
            --dark-slate: #495A6D;
            --deep-navy-purple: #43425F;
            --pale-teal: #B3D1D0;
            --peach: #E8AC92;
            --muted-lilac: #B9A5CF;
            --card: #FFFFFF;
            --paper: #EAF6FF;
            --ink: #43425F;
            --ink-soft: #617086;
            --navy: #43425F;
            --navy-soft: #617086;
            --brass: #E8AC92;
            --brass-strong: #F2967E;
            --teal: #98D8BF;
            --teal-soft: rgba(152, 216, 191, 0.22);
            --coral-soft: rgba(242, 150, 126, 0.18);
            --line: #D4D5DF;
        }

        /* ---------- base ---------- */
        .stApp { background: linear-gradient(135deg, var(--baby-blue) 0%, #F7FBFF 100%); }
        html, body, p, span, div, label, li {
            font-family: 'Source Sans 3', sans-serif;
            color: var(--ink);
        }
        h1, h2, h3, h4 {
            font-family: 'Fraunces', serif;
            color: var(--navy);
        }
        [data-testid="stHeader"] { background: transparent; }
        a:focus-visible, button:focus-visible {
            outline: 2px solid var(--brass);
            outline-offset: 2px;
        }

        /* ---------- sidebar ---------- */
        [data-testid="stSidebar"] { background: linear-gradient(180deg, var(--deep-navy-purple) 0%, var(--slate) 100%); }
        [data-testid="stSidebar"] > div { padding-top: 1.6rem; }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--paper) !important;
        }
        [data-testid="stSidebar"] button {
            color: var(--navy) !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: rgba(255,255,255,0.10) !important;
            border: 1px dashed rgba(234,246,255,0.55) !important;
            border-radius: 10px !important;
        }
        .cc-sidebar-title {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 1.45rem;
            color: var(--paper) !important;
            margin-bottom: 0.15rem;
        }
        .cc-sidebar-text {
            font-size: 0.86rem;
            color: rgba(247,242,231,0.72) !important;
            margin-bottom: 1.1rem;
        }
        .cc-sidebar-divider {
            height: 1px;
            background: rgba(247,242,231,0.18);
            margin: 1.5rem 0 1.1rem;
        }
        .cc-legend-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--peach) !important;
            margin-bottom: 0.6rem;
        }
        .cc-legend-row {
            font-size: 0.82rem;
            margin-bottom: 0.45rem;
            color: rgba(247,242,231,0.85) !important;
        }
        .cc-footnote {
            font-size: 0.75rem;
            color: rgba(247,242,231,0.55) !important;
            line-height: 1.5;
            margin-top: 1.2rem;
        }

        /* ---------- hero ---------- */
        .cc-hero {
            background: linear-gradient(135deg, var(--deep-navy-purple), var(--periwinkle));
            border: 1px solid rgba(143,161,224,0.35);
            border-radius: 14px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 0.4rem;
        }
        .cc-hero-eyebrow {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.28em;
            color: var(--brass);
            margin-bottom: 0.7rem;
            text-transform: uppercase;
        }
        .cc-hero-title {
            font-family: 'Fraunces', serif;
            font-weight: 900;
            font-size: 2.6rem;
            line-height: 1.1;
            color: #FFFFFF;
            margin-bottom: 0.7rem;
        }
        .cc-hero-tag {
            font-size: 1.02rem;
            color: rgba(255,255,255,0.88);
            max-width: 660px;
            line-height: 1.6;
        }
        .cc-divider {
            height: 1px;
            background: repeating-linear-gradient(90deg, var(--mint) 0 12px, transparent 12px 24px);
            opacity: 0.55;
            margin: 1.7rem 0 1.8rem;
        }

        /* ---------- generic panel ---------- */
        .cc-panel {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(212,213,223,0.9);
            border-radius: 12px;
            padding: 1.25rem 1.6rem;
            margin-bottom: 1.3rem;
        }
        .cc-panel-eyebrow {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.22em;
            color: var(--brass-strong);
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .cc-heading-value {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 1.55rem;
            color: var(--navy);
            margin-bottom: 0.25rem;
        }
        .cc-panel-sub {
            color: var(--ink-soft);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        /* ---------- empty state ---------- */
        .cc-empty-state {
            background: var(--card);
            border: 1px dashed var(--line);
            border-radius: 12px;
            padding: 2.6rem 2rem;
            text-align: center;
        }
        .cc-empty-state-title {
            font-family: 'Fraunces', serif;
            font-size: 1.35rem;
            color: var(--navy);
            margin-bottom: 0.5rem;
        }
        .cc-empty-state-text {
            color: var(--ink-soft);
            font-size: 0.95rem;
            max-width: 420px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ---------- job cards ---------- */
        .cc-job-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(212,213,223,0.9);
            border-radius: 12px;
            padding: 1.5rem 1.7rem;
            margin-bottom: 1.3rem;
        }
        .cc-job-rank {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.28em;
            color: var(--brass-strong);
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .cc-job-title {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 1.45rem;
            color: var(--navy);
            margin: 0.1rem 0 0.15rem;
            line-height: 1.25;
        }
        .cc-job-company {
            color: var(--ink-soft);
            font-size: 0.95rem;
            margin-bottom: 0.7rem;
        }
        .cc-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 1rem;
        }
        .cc-meta-pill {
            font-size: 0.78rem;
            padding: 0.22rem 0.7rem;
            border-radius: 999px;
            background: rgba(234,246,255,0.75);
            border: 1px solid var(--line);
            color: var(--ink-soft);
            white-space: nowrap;
        }
        .cc-job-body {
            display: flex;
            gap: 1.8rem;
            flex-wrap: wrap;
            border-top: 1px dashed var(--line);
            padding-top: 1.1rem;
        }
        .cc-gauge-col {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
            min-width: 170px;
        }
        .cc-bars { width: 100%; }
        .cc-bar-row { margin-bottom: 0.55rem; }
        .cc-bar-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.74rem;
            color: var(--ink-soft);
            margin-bottom: 0.22rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .cc-bar {
            height: 6px;
            background: var(--line);
            border-radius: 4px;
            overflow: hidden;
        }
        .cc-bar-fill { height: 100%; border-radius: 4px; }
        .cc-skills-col {
            flex: 1;
            min-width: 260px;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }
        .cc-skills-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--ink-soft);
            margin-bottom: 0.4rem;
        }
        .cc-chip {
            display: inline-block;
            padding: 0.18rem 0.65rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
        }
        .cc-chip-found {
            background: var(--teal-soft);
            color: #287765;
            border: 1px solid rgba(152,216,191,0.8);
        }
        .cc-chip-missing {
            background: var(--coral-soft);
            color: #D76F5F;
            border: 1px dashed rgba(242,150,126,0.85);
        }
        .cc-empty-chip {
            color: var(--ink-soft);
            font-size: 0.85rem;
            font-style: italic;
        }
        .cc-job-footer { margin-top: 1.2rem; }
        .cc-link-btn {
            display: inline-block;
            padding: 0.45rem 1.15rem;
            background: var(--deep-navy-purple);
            color: #FFFFFF !important;
            border-radius: 999px;
            font-size: 0.85rem;
            text-decoration: none;
            font-weight: 600;
            transition: background 0.15s ease;
        }
        .cc-link-btn:hover { background: var(--brass-strong); }

        /* ---------- compass gauge ---------- */
        .cc-gauge-ring {
            width: 96px;
            height: 96px;
            border-radius: 50%;
            background: conic-gradient(var(--coral) calc(var(--pct) * 1%), var(--lavender-grey) 0);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .cc-gauge-inner {
            width: 74px;
            height: 74px;
            border-radius: 50%;
            background: var(--card);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .cc-gauge-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1.15rem;
            color: var(--navy);
        }
        .cc-gauge-label {
            font-size: 0.58rem;
            letter-spacing: 0.2em;
            color: var(--ink-soft);
            margin-top: 0.1rem;
        }

        /* ---------- buttons & widgets ---------- */
        .stButton > button {
            background: linear-gradient(135deg, var(--coral), var(--peach));
            color: #FFFFFF;
            border: none;
            border-radius: 999px;
            padding: 0.55rem 1.7rem;
            font-weight: 600;
            font-family: 'Source Sans 3', sans-serif;
        }
        .stButton > button:hover { background: var(--brass-strong); color: #FFFFFF; }

        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            background: var(--card);
        }

        /* ---------- footer ---------- */
        .cc-footer {
            margin-top: 2.4rem;
            padding-top: 1.2rem;
            border-top: 1px dashed var(--line);
            font-size: 0.78rem;
            color: var(--ink-soft);
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_resources():
    df = pd.read_csv("linkedin_jobs_clustered.csv")

    sbert_model = SentenceTransformer("career_sbert_finetuned")
    kmeans_model = joblib.load("kmeans_model.pkl")

    job_embeddings = np.load("job_sbert_finetuned_embeddings.npy")
    job_embeddings_norm = normalize(job_embeddings)

    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_job_text"].fillna(""))

    return df, sbert_model, kmeans_model, job_embeddings_norm, tfidf_vectorizer, tfidf_matrix


df, sbert_model, kmeans_model, job_embeddings_norm, tfidf_vectorizer, tfidf_matrix = load_resources()


# -----------------------------
# Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# -----------------------------
# Resume extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    return text


def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    else:
        return ""


# -----------------------------
# Skill gap analysis
# -----------------------------
skill_keywords = [
    "python", "java", "javascript", "sql", "excel", "tableau", "power bi",
    "machine learning", "deep learning", "nlp", "data analysis",
    "data visualization", "statistics", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "aws", "azure", "docker", "kubernetes",
    "react", "node.js", "html", "css", "git", "github",
    "communication", "leadership", "project management", "agile",
    "marketing", "seo", "sales", "recruitment", "hr", "finance"
]

def extract_skills(text):
    text = str(text).lower()
    found = []

    for skill in skill_keywords:
        if skill in text:
            found.append(skill)

    return sorted(list(set(found)))


def get_skill_gap(resume_text, job_description):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))

    matched = sorted(list(resume_skills.intersection(job_skills)))
    missing = sorted(list(job_skills - resume_skills))

    return matched, missing


# -----------------------------
# Hybrid recommendation function
# -----------------------------
def recommend_jobs_hybrid(resume_text, top_n=10):
    clean_resume = preprocess_text(resume_text)

    resume_tfidf = tfidf_vectorizer.transform([clean_resume])
    tfidf_scores = cosine_similarity(resume_tfidf, tfidf_matrix).flatten()

    resume_embedding = sbert_model.encode([resume_text])
    resume_embedding_norm = normalize(resume_embedding)

    sbert_scores = cosine_similarity(
        resume_embedding_norm,
        job_embeddings_norm
    ).flatten()

    resume_cluster = kmeans_model.predict(resume_embedding_norm)[0]

    cluster_indices = df[df["cluster"] == resume_cluster].index

    hybrid_scores = (0.4 * tfidf_scores) + (0.6 * sbert_scores)

    cluster_scores = hybrid_scores[cluster_indices]
    top_cluster_positions = cluster_scores.argsort()[::-1][:top_n]
    top_indices = cluster_indices[top_cluster_positions]

    results = df.loc[top_indices].copy()

    results["tfidf_score"] = tfidf_scores[top_indices]
    results["sbert_score"] = sbert_scores[top_indices]
    results["hybrid_score"] = hybrid_scores[top_indices]

    matched_skills_list = []
    missing_skills_list = []

    for idx in results.index:
        matched, missing = get_skill_gap(
            resume_text,
            df.loc[idx, "description"]
        )

        matched_skills_list.append(", ".join(matched) if matched else "None")
        missing_skills_list.append(", ".join(missing) if missing else "None")

    results["matched_skills"] = matched_skills_list
    results["missing_skills"] = missing_skills_list

    return results, resume_cluster

# -----------------------------
# TF-IDF Only Recommendation
# -----------------------------
def recommend_jobs_tfidf(resume_text, top_n=10):

    clean_resume = preprocess_text(resume_text)

    resume_tfidf = tfidf_vectorizer.transform([clean_resume])

    tfidf_scores = cosine_similarity(
        resume_tfidf,
        tfidf_matrix
    ).flatten()

    top_indices = tfidf_scores.argsort()[::-1][:top_n]

    results = df.loc[top_indices].copy()

    results["score"] = tfidf_scores[top_indices]

    return results


# -----------------------------
# SBERT Only Recommendation
# -----------------------------
def recommend_jobs_sbert(resume_text, top_n=10):

    resume_embedding = sbert_model.encode([resume_text])

    resume_embedding_norm = normalize(resume_embedding)

    sbert_scores = cosine_similarity(
        resume_embedding_norm,
        job_embeddings_norm
    ).flatten()

    top_indices = sbert_scores.argsort()[::-1][:top_n]

    results = df.loc[top_indices].copy()

    results["score"] = sbert_scores[top_indices]

    return results

# -----------------------------
# Card rendering
# -----------------------------
def _pct(value):
    return max(0, min(100, round(float(value) * 100)))


def render_job_card(rank, row):
    hybrid_pct = _pct(row["hybrid_score"])
    keyword_pct = _pct(row["tfidf_score"])
    meaning_pct = _pct(row["sbert_score"])

    meta_items = []
    if pd.notna(row.get("location")):
        meta_items.append(f"📍 {esc(str(row['location']))}")
    if pd.notna(row.get("formatted_work_type")):
        meta_items.append(f"🗂 {esc(str(row['formatted_work_type']))}")
    if pd.notna(row.get("formatted_experience_level")):
        meta_items.append(f"🎯 {esc(str(row['formatted_experience_level']))}")
    salary = row.get("normalized_salary", -1)
    if pd.notna(salary) and salary != -1:
        meta_items.append(f"💰 {salary:,.0f}")

    meta_html = "".join(f"<span class='cc-meta-pill'>{m}</span>" for m in meta_items)

    matched = row["matched_skills"]
    missing = row["missing_skills"]

    if matched != "None":
        matched_html = "".join(
            f"<span class='cc-chip cc-chip-found'>{esc(s.strip())}</span>"
            for s in matched.split(",")
        )
    else:
        matched_html = "<span class='cc-empty-chip'>No overlapping skills detected</span>"

    if missing != "None":
        missing_html = "".join(
            f"<span class='cc-chip cc-chip-missing'>{esc(s.strip())}</span>"
            for s in missing.split(",")
        )
    else:
        missing_html = "<span class='cc-empty-chip'>Nothing extra to add — full coverage</span>"

    link_html = ""
    if pd.notna(row.get("job_posting_url")):
        link_html = (
            f"<div class='cc-job-footer'>"
            f"<a class='cc-link-btn' href='{esc(str(row['job_posting_url']), quote=True)}' "
            f"target='_blank' rel='noopener noreferrer'>View job posting →</a>"
            f"</div>"
        )

    card_html = f"""
    <div class="cc-job-card">
        <div class="cc-job-rank">No. {rank:02d}</div>
        <div class="cc-job-title">{esc(str(row['title']))}</div>
        <div class="cc-job-company">{esc(str(row['company_name']))}</div>
        <div class="cc-meta-row">{meta_html}</div>
        <div class="cc-job-body">
            <div class="cc-gauge-col">
                <div class="cc-gauge-ring" style="--pct:{hybrid_pct}">
                    <div class="cc-gauge-inner">
                        <span class="cc-gauge-value">{hybrid_pct}%</span>
                        <span class="cc-gauge-label">MATCH</span>
                    </div>
                </div>
                <div class="cc-bars">
                    <div class="cc-bar-row">
                        <div class="cc-bar-label"><span>Keyword overlap</span><span>{keyword_pct}%</span></div>
                        <div class="cc-bar"><div class="cc-bar-fill" style="width:{keyword_pct}%; background: var(--periwinkle);"></div></div>
                    </div>
                    <div class="cc-bar-row">
                        <div class="cc-bar-label"><span>Meaning overlap</span><span>{meaning_pct}%</span></div>
                        <div class="cc-bar"><div class="cc-bar-fill" style="width:{meaning_pct}%; background: var(--mint);"></div></div>
                    </div>
                </div>
            </div>
            <div class="cc-skills-col">
                <div>
                    <div class="cc-skills-title">On your resume</div>
                    <div>{matched_html}</div>
                </div>
                <div>
                    <div class="cc-skills-title">Worth adding</div>
                    <div>{missing_html}</div>
                </div>
            </div>
        </div>
        {link_html}
    </div>
    """
    return card_html



# -----------------------------
# Model comparison and evaluation helpers
# -----------------------------
def prepare_display_df(results, model_type):
    """Return a compact table for Streamlit display."""
    cols = ["title", "company_name", "location"]

    if model_type == "tfidf":
        display_df = results[cols + ["score"]].copy()
        display_df = display_df.rename(columns={"score": "tfidf_score"})

    elif model_type == "sbert":
        display_df = results[cols + ["score"]].copy()
        display_df = display_df.rename(columns={"score": "sbert_score"})

    else:
        display_df = results[
            cols + [
                "tfidf_score",
                "sbert_score",
                "hybrid_score",
                "matched_skills",
                "missing_skills",
                "job_posting_url",
            ]
        ].copy()

    return display_df


def show_model_results_table(results, model_type):
    """Display recommendation results in a clean table."""
    display_df = prepare_display_df(results, model_type)
    st.dataframe(display_df, use_container_width=True)


def skill_match_ratio_for_job(resume_text, job_description):
    """Automatic skill-overlap score between 0 and 1."""
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))

    if len(job_skills) == 0:
        return 0.0

    return len(resume_skills.intersection(job_skills)) / len(job_skills)


def automatic_model_score(results, resume_text, resume_embedding_norm, resume_cluster):
    """
    Automatic proxy evaluation for a recommendation list.

    This is not ground-truth accuracy. Since the dataset has no real resume-job labels,
    the app compares models using signals it can compute itself:
    1. Semantic similarity to the uploaded resume
    2. Skill overlap between resume and job description
    3. Cluster consistency with the resume's predicted career cluster
    """
    if len(results) == 0:
        return {
            "Semantic Similarity": 0.0,
            "Skill Match": 0.0,
            "Cluster Consistency": 0.0,
            "Automatic Overall Score": 0.0,
        }

    result_indices = results.index.to_numpy()

    semantic_scores = cosine_similarity(
        resume_embedding_norm,
        job_embeddings_norm[result_indices]
    ).flatten()
    semantic_mean = float(np.clip(np.mean(semantic_scores), 0, 1))

    skill_scores = []
    for idx in result_indices:
        skill_scores.append(
            skill_match_ratio_for_job(
                resume_text,
                df.loc[idx, "description"]
            )
        )
    skill_mean = float(np.mean(skill_scores)) if skill_scores else 0.0

    cluster_values = df.loc[result_indices, "cluster"].to_numpy()
    cluster_consistency = float(np.mean(cluster_values == resume_cluster))

    overall_score = (
        0.50 * semantic_mean
        + 0.30 * skill_mean
        + 0.20 * cluster_consistency
    )

    return {
        "Semantic Similarity": semantic_mean,
        "Skill Match": skill_mean,
        "Cluster Consistency": cluster_consistency,
        "Automatic Overall Score": overall_score,
    }


def show_automatic_model_comparison(resume_text, resume_cluster, tfidf_results, sbert_results, hybrid_results):
    """Compare TF-IDF, SBERT, and Hybrid automatically without manual labels."""
    resume_embedding = sbert_model.encode([resume_text])
    resume_embedding_norm = normalize(resume_embedding)

    rows = []
    model_map = {
        "TF-IDF Baseline": tfidf_results,
        "Fine-tuned SBERT": sbert_results,
        "Hybrid Model": hybrid_results,
    }

    for model_name, results in model_map.items():
        scores = automatic_model_score(
            results,
            resume_text,
            resume_embedding_norm,
            resume_cluster
        )
        rows.append({"Model": model_name, **scores})

    comparison_df = pd.DataFrame(rows)

    st.markdown("## 📊 Automatic Model Comparison")
    st.caption(
        "This comparison does not use manual labels. It is an automatic proxy evaluation based on "
        "semantic similarity, skill overlap, and cluster consistency for the uploaded resume."
    )

    st.dataframe(comparison_df, use_container_width=True)

    chart_df = comparison_df.set_index("Model")[["Automatic Overall Score"]]
    st.bar_chart(chart_df)

    best_row = comparison_df.sort_values(
        by="Automatic Overall Score",
        ascending=False
    ).iloc[0]

    st.success(
        f"Best automatic performer for this resume: **{best_row['Model']}** "
        f"with overall score **{best_row['Automatic Overall Score']:.3f}**"
    )

    with st.expander("How this automatic score is calculated"):
        st.markdown(
            """
            **Automatic Overall Score** is a proxy score because the dataset does not contain true resume-job relevance labels.

            It combines:

            - **50% Semantic Similarity**: how close the resume and recommended jobs are using fine-tuned SBERT embeddings.
            - **30% Skill Match**: how many required job skills are found in the resume.
            - **20% Cluster Consistency**: how many recommended jobs belong to the same career cluster as the resume.

            This lets the system compare TF-IDF, SBERT, and Hybrid recommendations automatically without manual selection.
            """
        )


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<div class='cc-sidebar-title'>🧭 CareerCompass</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='cc-sidebar-text'>Upload a resume to discover roles that best match your skills and meaning.</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Resume", type=["pdf", "txt"], label_visibility="collapsed")

    top_n = st.slider("Number of recommendations", min_value=5, max_value=15, value=10)

    st.markdown("<div class='cc-sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='cc-legend-title'>Reading the results</div>
        <div class='cc-legend-row'><span class='cc-chip cc-chip-found'>skill</span> already on your resume</div>
        <div class='cc-legend-row'><span class='cc-chip cc-chip-missing'>skill</span> the role wants but you don't list</div>
        <div class='cc-legend-row'>The dial shows a hybrid match score — how close your resume is to each role overall.</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class='cc-footnote'>
        Hybrid score = 40% keyword overlap (TF-IDF) + 60% semantic similarity
        (a fine-tuned SBERT model). Roles are then narrowed to the career
        cluster your resume is closest to.
        </p>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="cc-hero">
        <div class="cc-hero-eyebrow">Hybrid Resume–Job Matching</div>
        <div class="cc-hero-title">🧭 CareerCompass</div>
        <div class="cc-hero-tag">
            Upload a resume to get ranked job recommendations using TF-IDF, Career-SBERT,
            cluster filtering, and skill gap analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='cc-divider'></div>", unsafe_allow_html=True)


# -----------------------------
# Main flow
# -----------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
    st.session_state.resume_cluster = None
    st.session_state.tfidf_results = None
    st.session_state.sbert_results = None

if uploaded_file is None:
    st.markdown(
        """
        <div class="cc-empty-state">
            <div class="cc-empty-state-title">No resume uploaded yet</div>
            <div class="cc-empty-state-text">
                Use the panel on the left to upload a PDF or TXT resume.
                The app will process it, identify its closest career cluster,
                and return the most relevant job matches.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    resume_text = extract_resume_text(uploaded_file)

    if len(resume_text.strip()) == 0:

        st.error(
            "Could not extract text from the uploaded file. "
            "Try a different PDF or a plain .txt file."
        )

    else:

        st.success(
            "Resume read successfully. Ready to find your matches."
        )

        with st.expander("Preview extracted resume text"):
            st.text(resume_text[:3000])

        if st.button("Find my matches", type="primary"):

            with st.spinner("Finding your best matches..."):

                # Hybrid recommendations
                recommendations, resume_cluster = recommend_jobs_hybrid(
                    resume_text,
                    top_n=top_n
                )

                # TF-IDF recommendations
                tfidf_results = recommend_jobs_tfidf(
                    resume_text,
                    top_n=top_n
                )

                # SBERT recommendations
                sbert_results = recommend_jobs_sbert(
                    resume_text,
                    top_n=top_n
                )

            st.session_state.recommendations = recommendations
            st.session_state.resume_cluster = resume_cluster
            st.session_state.tfidf_results = tfidf_results
            st.session_state.sbert_results = sbert_results

# --------------------------------------------------
# HYBRID RESULTS
# --------------------------------------------------
if st.session_state.recommendations is not None:

    st.markdown(
        f"""
        <div class="cc-panel">
            <div class="cc-panel-eyebrow">Nearest career cluster</div>
            <div class="cc-heading-value">
                Cluster {st.session_state.resume_cluster}
            </div>
            <div class="cc-panel-sub">
                Your resume is assigned to this job cluster first.
                Recommendations below are filtered within that cluster and ranked
                using the hybrid score — highest first.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Top Hybrid Recommendations")

    for i, (_, row) in enumerate(
        st.session_state.recommendations.iterrows(),
        start=1
    ):
        st.markdown(
            render_job_card(i, row),
            unsafe_allow_html=True
        )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------
    if (
        st.session_state.tfidf_results is not None
        and st.session_state.sbert_results is not None
    ):
        st.markdown("---")
        st.markdown("## 📊 Compare All Three Models")
        st.caption(
            "These results are generated for the same uploaded resume. "
            "TF-IDF uses keyword overlap, SBERT uses semantic similarity, "
            "and Hybrid combines both after cluster filtering."
        )

        tab1, tab2, tab3 = st.tabs([
            "TF-IDF Baseline",
            "Fine-tuned SBERT",
            "Hybrid Model",
        ])

        with tab1:
            st.markdown("### TF-IDF Baseline Recommendations")
            show_model_results_table(
                st.session_state.tfidf_results,
                model_type="tfidf"
            )

        with tab2:
            st.markdown("### Fine-tuned SBERT Recommendations")
            show_model_results_table(
                st.session_state.sbert_results,
                model_type="sbert"
            )

        with tab3:
            st.markdown("### Hybrid Recommendations")
            show_model_results_table(
                st.session_state.recommendations,
                model_type="hybrid"
            )

        show_automatic_model_comparison(
            resume_text=resume_text,
            resume_cluster=st.session_state.resume_cluster,
            tfidf_results=st.session_state.tfidf_results,
            sbert_results=st.session_state.sbert_results,
            hybrid_results=st.session_state.recommendations,
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="cc-footer">
        CareerCompass &mdash; hybrid TF-IDF + Career-SBERT similarity,
        filtered by K-Means clustering and supported by skill gap analysis.
    </div>
    """,
    unsafe_allow_html=True,
)