# ============================================================
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  Project 3  : AI Recommendation Logic
#  Algorithm  : Content-Based Filtering
#               TF-IDF Vectorization + Cosine Similarity
#  Capstone   : Tech Stack Recommender
#  Pipeline   : Ingest → Score → Sort → Filter (Top-N)
# ============================================================
#
#  HOW TO RUN:
#  pip install scikit-learn pandas numpy
#  python recommender.py
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── DATASET PATH ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "raw_skills.csv")


# ── STEP 1: LOAD KNOWLEDGE BASE ──────────────────────────────
# Each row = one job role with required skills as
# a space-separated string (the "document" in TF-IDF terms)

def load_knowledge_base(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df.columns  = df.columns.str.strip()
    df["skills"] = df["skills"].str.lower().str.strip()

    print("=" * 55)
    print("  PROJECT 3 — AI RECOMMENDATION LOGIC")
    print("  Tech Stack Recommender | Content-Based Filtering")
    print("=" * 55)
    print(f"\n[KNOWLEDGE BASE]")
    print(f"  Job Roles Loaded : {len(df)}")
    print(f"  Sample Roles     : {list(df['job_role'].head(5))}")
    return df


# ── STEP 2: TF-IDF VECTORIZATION ─────────────────────────────
# Converts skill strings into weighted numerical vectors.
#
# TF  = how often skill appears in that role's document
# IDF = penalizes common skills (python) vs rare ones (solidity)
#
# Solves Binary Overlap Problem:
# Two roles with 3 overlapping skills are NOT equal
# if one has specific rare skills and other has generic ones

def build_tfidf_matrix(df):
    vectorizer  = TfidfVectorizer(
        tokenizer = lambda x: x.split(),
        lowercase = True
    )
    tfidf_matrix = vectorizer.fit_transform(df["skills"])

    print(f"\n[TF-IDF VECTORIZATION]")
    print(f"  Matrix Shape     : {tfidf_matrix.shape}")
    print(f"  Job Roles        : {tfidf_matrix.shape[0]}")
    print(f"  Unique Skills    : {tfidf_matrix.shape[1]} dimensions")
    print(f"  Method           : TF × IDF weighting")

    return tfidf_matrix, vectorizer


# ── STEP 3: BUILD USER PROFILE VECTOR ────────────────────────
# User's skills → same TF-IDF vocabulary space as job roles
# MUST use the same fitted vectorizer (shared vocabulary)
#
# Cold Start: zero vector if no skills match → detected & warned

def build_user_vector(user_skills: list, vectorizer):
    user_text   = " ".join(s.lower().strip() for s in user_skills)
    user_vector = vectorizer.transform([user_text])

    if user_vector.nnz == 0:
        print("\n[!] COLD START DETECTED")
        print("    None of your skills matched the vocabulary.")
        print("    Try: python, sql, docker, ml, aws, react, java")
        return None

    print(f"\n[USER VECTOR]")
    print(f"  Skills Provided  : {user_skills}")
    print(f"  Active Dimensions: {user_vector.nnz} matching features")
    return user_vector


# ── STEP 4: COSINE SIMILARITY SCORING ────────────────────────
# Measures ANGLE between vectors — magnitude invariant
# A short CV and long CV pointing same skill direction = equal score
#
# Score 1.0 → perfect match
# Score 0.0 → no overlap
# TF-IDF gives non-negative values → range naturally 0 to 1

def score_all_roles(user_vector, tfidf_matrix, df):
    scores          = cosine_similarity(user_vector, tfidf_matrix).flatten()
    df              = df.copy()
    df["score"]     = scores
    df["match_pct"] = (scores * 100).round(2)
    return df


# ── STEP 5: SORT + FILTER (Top-N) ────────────────────────────
# Step 3: Sort descending by score → best match first
# Step 4: Truncate to Top-N → prevents choice overload

def get_top_n(scored_df, n=3):
    top = scored_df.sort_values("score", ascending=False).head(n)
    return top.reset_index(drop=True)


# ── OUTPUT DISPLAY ────────────────────────────────────────────

def display_recommendations(top_n, user_skills):
    medals = ["🥇", "🥈", "🥉"]
    print(f"\n{'='*55}")
    print(f"  TOP {len(top_n)} CAREER RECOMMENDATIONS")
    print(f"  Your Skills : {', '.join(user_skills)}")
    print(f"{'='*55}")

    for i, row in top_n.iterrows():
        medal = medals[i] if i < 3 else f"#{i+1}"
        bar   = "█" * int(row["score"] * 30)
        print(f"\n  {medal}  {row['job_role']}")
        print(f"       Match  : {row['match_pct']}%  {bar}")
        print(f"       Skills : {row['skills'][:65]}...")

    print(f"\n{'='*55}")
    return top_n


# ── USER INPUT COLLECTION ─────────────────────────────────────
# Minimum 3 inputs required (spec: sufficient data density)

def collect_user_skills(interactive=True, prefilled=None):
    if not interactive and prefilled:
        print(f"\n[INPUT] Skills from pipeline: {prefilled}")
        return prefilled

    print(f"\n[INPUT] Enter your skills (minimum 3 required)")
    print(f"  Examples: python, sql, docker, ml, aws, react\n")

    skills = []
    while len(skills) < 3:
        try:
            skill = input(f"  Skill {len(skills)+1} [required]: ").strip()
            if skill:
                skills.append(skill.lower())
        except (KeyboardInterrupt, EOFError):
            break

    # Optional extras
    if interactive:
        print(f"  (Press Enter to finish adding skills)")
        while True:
            try:
                extra = input(f"  Skill {len(skills)+1} [optional]: ").strip()
                if extra == "":
                    break
                skills.append(extra.lower())
            except (KeyboardInterrupt, EOFError):
                break

    print(f"\n  ✓ {len(skills)} skills collected: {skills}")
    return skills


# ── MAIN PIPELINE ─────────────────────────────────────────────

def run_recommender(interactive=True, prefilled_skills=None, top_n=3):
    """
    Full 4-step recommendation pipeline.
    interactive=False + prefilled_skills → used by main.py
    Returns: top_n DataFrame
    """
    df                        = load_knowledge_base()
    tfidf_matrix, vectorizer  = build_tfidf_matrix(df)
    user_skills               = collect_user_skills(interactive, prefilled_skills)

    if not user_skills:
        print("[!] No skills provided. Exiting.")
        return None

    user_vector = build_user_vector(user_skills, vectorizer)
    if user_vector is None:
        return None

    scored_df   = score_all_roles(user_vector, tfidf_matrix, df)
    top_results = get_top_n(scored_df, n=top_n)
    display_recommendations(top_results, user_skills)

    return top_results


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    run_recommender(interactive=True, top_n=3)
