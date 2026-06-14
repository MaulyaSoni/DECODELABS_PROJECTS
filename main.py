# ============================================================
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  MEGA PROJECT : DecodeLabs AI Suite
#  Pipeline     : P1 Chatbot → P2 Classification
#                 → P3 Recommendation → P4 Recognition
#
#  HOW TO RUN:
#  pip install -r requirements.txt
#  python main.py
# ============================================================

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "project1_chatbot"))
sys.path.insert(0, os.path.join(BASE_DIR, "project2_classification"))
sys.path.insert(0, os.path.join(BASE_DIR, "project3_recommendation"))
sys.path.insert(0, os.path.join(BASE_DIR, "project4_recognition"))

from classifier  import run_classifier
from recommender import run_recommender
from recognition import run_recognition

# ── SKILL KEYWORDS VOCABULARY ────────────────────────────────
SKILL_KEYWORDS = [
    "python","java","javascript","c","kotlin","swift","typescript","r",
    "sql","mysql","postgresql","mongodb","nosql","firebase",
    "machine_learning","ml","deep_learning","ai","nlp",
    "tensorflow","pytorch","keras","scikit_learn","sklearn",
    "pandas","numpy","matplotlib","seaborn","scipy",
    "docker","kubernetes","aws","azure","gcp","cloud","terraform",
    "react","nodejs","html","css","vuejs","angular",
    "linux","bash","git","ci_cd","devops","mlops",
    "data_analysis","statistics","visualization","tableau","power_bi",
    "blockchain","solidity","web3","security","networking",
    "android","ios","mobile","unity","game","c_sharp","c_plus_plus",
    "transformers","bert","gpt","llm","generative_ai","prompt_engineering",
    "automation","selenium","testing","rest_api","microservices",
    "spark","hadoop","kafka","etl","data_pipelines","airflow",
    "computer_vision","opencv","cnn","object_detection","ocr",
]

SKILL_QUESTIONS = [
    "What programming languages do you know?\n"
    "  (e.g. Python, Java, JavaScript, C++)",
    "What frameworks or tools have you worked with?\n"
    "  (e.g. TensorFlow, React, Docker, Django, Flask)",
    "What domains interest you most?\n"
    "  (e.g. ML, Web Dev, Cloud, DevOps, Data Science, Computer Vision)",
]


# ── STAGE 1: CHATBOT ─────────────────────────────────────────
def chatbot_stage() -> tuple:
    print("\n" + "="*55)
    print("  STAGE 1 — CHATBOT (Project 1)")
    print("="*55)
    print(f"\nDBot: Hi! I'm DBot from DecodeLabs Batch 2026.")
    print(f"      I'll find the best career path for you.")
    print(f"      Answer 3 quick questions!\n")

    all_skills = []
    history    = []

    for i, question in enumerate(SKILL_QUESTIONS):
        print(f"DBot: Q{i+1}) {question}")
        try:
            user_input = input("You : ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input:
            continue
        history.append({"role": "user", "content": user_input})
        found = []
        text  = user_input.lower()
        for skill in SKILL_KEYWORDS:
            if skill.replace("_", " ") in text or skill in text:
                found.append(skill)
        if not found:
            words = [w.strip(",.!?()") for w in user_input.split() if len(w) > 2]
            found = words[:4]
        all_skills.extend(found)
        print(f"DBot: Got it! Captured → {found}\n")

    seen = set()
    unique_skills = []
    for s in all_skills:
        if s not in seen:
            seen.add(s)
            unique_skills.append(s)

    print(f"DBot: Your skill profile → {unique_skills}\n")
    return unique_skills, history


# ── STAGE 2: CLASSIFICATION ───────────────────────────────────
def classification_stage(skills: list) -> dict:
    print("\n" + "="*55)
    print("  STAGE 2 — CLASSIFICATION (Project 2)")
    print("="*55)
    print("\n[P2] Running Iris KNN Pipeline...\n")
    run_classifier(interactive=False)

    skill_set     = set(s.lower() for s in skills)
    ml_skills     = {"python","ml","machine_learning","deep_learning","tensorflow","pytorch","scikit_learn","numpy","pandas","ai","nlp","computer_vision","ocr"}
    web_skills    = {"javascript","react","nodejs","html","css","typescript","vuejs","angular"}
    data_skills   = {"sql","pandas","statistics","data_analysis","tableau","power_bi","visualization"}
    devops_skills = {"docker","kubernetes","aws","azure","ci_cd","linux","bash","terraform","gcp"}

    scores   = {"AI/ML": len(skill_set & ml_skills), "Web Dev": len(skill_set & web_skills), "Data": len(skill_set & data_skills), "DevOps": len(skill_set & devops_skills)}
    dominant = max(scores, key=scores.get) if any(scores.values()) else "General"
    total    = len(skill_set)
    tier     = "Senior" if total >= 8 else "Intermediate" if total >= 5 else "Junior" if total >= 2 else "Beginner"

    profile = {"total": total, "dominant": dominant, "tier": tier, "scores": scores}
    print(f"\n[USER PROFILE] Tier: {tier} | Domain: {dominant} | Skills: {total}")
    return profile


# ── STAGE 3: RECOMMENDATION ───────────────────────────────────
def recommendation_stage(skills: list):
    print("\n" + "="*55)
    print("  STAGE 3 — RECOMMENDATION ENGINE (Project 3)")
    print("="*55)
    return run_recommender(interactive=False, prefilled_skills=skills, top_n=3)


# ── STAGE 4: RECOGNITION (Optional) ──────────────────────────
def recognition_stage():
    print("\n" + "="*55)
    print("  STAGE 4 — IMAGE/TEXT RECOGNITION (Project 4)")
    print("  Optional Mastery Phase")
    print("="*55)
    try:
        choice = input("\n  Run Project 4 Recognition demo? (y/n): ").strip().lower()
        if choice == "y":
            run_recognition(path_choice=1)
    except (KeyboardInterrupt, EOFError):
        pass


# ── FINAL SUMMARY ─────────────────────────────────────────────
def print_summary(skills, profile, recommendations):
    print("\n" + "█"*55)
    print("  PIPELINE COMPLETE — YOUR PERSONALIZED REPORT")
    print("  DecodeLabs AI Suite | Batch 2026")
    print("█"*55)
    print(f"\n  Skills Detected    : {skills}")
    print(f"  Readiness Tier     : {profile['tier']}")
    print(f"  Dominant Domain    : {profile['dominant']}")
    if recommendations is not None and len(recommendations) > 0:
        medals = ["🥇", "🥈", "🥉"]
        print(f"\n  Your Top Career Paths:")
        for i, row in recommendations.iterrows():
            m = medals[i] if i < 3 else f"#{i+1}"
            print(f"    {m}  {row['job_role']:<30} {row['match_pct']}% match")
    print(f"\n  Powered by DecodeLabs | www.decodelabs.tech")
    print("█"*55 + "\n")


# ── MEGA PIPELINE ─────────────────────────────────────────────
def run_mega_pipeline():
    print("\n" + "█"*55)
    print("  DECODELABS AI SUITE — MEGA PROJECT")
    print("  All 4 Projects | One Unified Pipeline")
    print("  Batch 2026 | Powered by DecodeLabs")
    print("█"*55)
    print("""
  FLOW:
  ┌──────────────────────┐
  │ P1: Chatbot          │ → Collects skills via conversation
  └────────┬─────────────┘
           ↓
  ┌──────────────────────┐
  │ P2: KNN Classifier   │ → Iris demo + User profile tier
  └────────┬─────────────┘
           ↓
  ┌──────────────────────┐
  │ P3: Recommender      │ → TF-IDF + Cosine → Top 3 careers
  └────────┬─────────────┘
           ↓
  ┌──────────────────────┐
  │ P4: Recognition      │ → OCR or Object Detection (optional)
  └──────────────────────┘
    """)

    try:
        input("  Press Enter to start...\n")
    except (KeyboardInterrupt, EOFError):
        return

    skills, history  = chatbot_stage()
    if not skills:
        skills = ["python", "machine_learning", "sql", "pandas"]

    profile          = classification_stage(skills)
    recommendations  = recommendation_stage(skills)
    recognition_stage()
    print_summary(skills, profile, recommendations)


if __name__ == "__main__":
    run_mega_pipeline()
