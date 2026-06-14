# ============================================================
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  Project 1  : Rule-Based AI Chatbot — VERSION 2 (HYBRID)
#  Type       : Rule-Based + Groq Llama 3.3 Fallback
#  Levels     : L1 Keyword Extraction
#               L2 Intent Grouping
#               L3 Regex Pattern Matching
#               L4 Groq Llama 3.3 Fallback (Free Tier)
# ============================================================
#
#  HOW TO RUN:
#  1. pip install groq
#  2. Get FREE API key → https://console.groq.com
#  3. Paste key in GROQ_API_KEY below
#  4. python chatbot_v2.py
#
#  FREE TIER LIMITS (Groq):
#  → 30 requests/minute
#  → No credit card needed
#  → Model: llama-3.3-70b-versatile
# ============================================================

import re
import os

# ── GROQ CONFIGURATION ────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.3-70b-versatile"   # Best free model on Groq
BOT_NAME      = "DBot"

SYSTEM_PROMPT = (
    f"You are {BOT_NAME}, a friendly AI assistant built by DecodeLabs "
    f"for their AI internship batch 2026 in Lucknow, India. "
    f"Keep responses concise (3-4 lines max), clear, and helpful. "
    f"Speak in plain conversational text without markdown or bullet points."
)


# ═══════════════════════════════════════════════════════════════
#  LEVEL 4 — GROQ LLAMA 3.3 FALLBACK
# ═══════════════════════════════════════════════════════════════

def init_groq():
    """Initialize Groq client. Returns client or None if key missing."""
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        # Quick test call to validate key
        return client
    except ImportError:
        print("[WARNING] groq package not installed. Run: pip install groq")
        return None
    except Exception as e:
        print(f"[WARNING] Groq init failed: {e}")
        return None


def ask_groq(client, conversation_history: list, user_input: str) -> str:
    """Send conversation + new input to Groq Llama 3.3."""
    try:
        # Build messages: system prompt + last 6 turns + new input
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in conversation_history[-6:]:
            role = turn["role"]
            # Groq uses "user" and "assistant" (not "model")
            if role == "model":
                role = "assistant"
            messages.append({"role": role, "content": turn["content"]})

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        err = str(e)
        if "429" in err:
            return "Rate limit hit. Wait 10 seconds and try again. (Groq free: 30 req/min)"
        if "401" in err or "403" in err:
            return "Invalid API key. Check your GROQ_API_KEY in chatbot_v2.py"
        return f"Groq error: {err[:80]}. Falling back to rule-based mode."


# ═══════════════════════════════════════════════════════════════
#  LEVEL 2 — INTENT GROUPS (Keyword Sets → Single Response)
# ═══════════════════════════════════════════════════════════════

INTENT_GROUPS = [
    {
        "intent"   : "greeting",
        "keywords" : ["hello","hi","hey","sup","greetings","good morning",
                      "good evening","good afternoon","what's up","whats up","howdy"],
        "response" : f"Hey! I'm {BOT_NAME}, your hybrid AI assistant. What's on your mind?"
    },
    {
        "intent"   : "farewell",
        "keywords" : ["bye","goodbye","see you","take care","later","cya",
                      "good night","goodnight"],
        "response" : "Goodbye! Keep building. See you next time. 🚀"
    },
    {
        "intent"   : "identity",
        "keywords" : ["who are you","what are you","your name","introduce yourself",
                      "tell me about yourself","are you a bot","are you human","are you ai"],
        "response" : (
            f"I'm {BOT_NAME} v2 — a hybrid AI chatbot built at DecodeLabs. "
            f"I use rule-based logic first (fast & deterministic), "
            f"then Groq Llama 3.3 as my fallback brain for unknown questions."
        )
    },
    {
        "intent"   : "capability",
        "keywords" : ["what can you do","help","commands","options",
                      "how do you work","what do you know","capabilities"],
        "response" : (
            "Here's what I can handle:\n"
            "  → Greetings and farewells\n"
            "  → Questions about AI, ML, LLMs, NLP\n"
            "  → 'What is [anything]?' — I'll explain it\n"
            "  → 'How to [do anything]?' — Groq handles it\n"
            "  → Jokes, motivation, time\n"
            "  → Any unknown question → Groq Llama 3.3 answers it!"
        )
    },
    {
        "intent"   : "creator",
        "keywords" : ["who made you","who built you","who created you",
                      "who programmed you","who coded you"],
        "response" : f"Built by a DecodeLabs AI intern, Batch 2026. Powered by rules + Llama 3.3."
    },
    {
        "intent"   : "ai_concept",
        "keywords" : ["what is ai","artificial intelligence","define ai","explain ai"],
        "response" : (
            "AI (Artificial Intelligence) is the simulation of human intelligence "
            "by machines — enabling them to learn, reason, and make decisions from data."
        )
    },
    {
        "intent"   : "ml_concept",
        "keywords" : ["what is ml","machine learning","define ml","explain machine learning"],
        "response" : (
            "ML (Machine Learning) is a subset of AI where systems automatically "
            "learn patterns from data without being explicitly programmed for each task."
        )
    },
    {
        "intent"   : "llm_concept",
        "keywords" : ["what is llm","large language model","what is gpt","what is gemini",
                      "what is claude","what is llama","generative ai"],
        "response" : (
            "LLMs are massive neural networks trained on text. "
            "They predict the next token to generate human-like responses. "
            "Examples: GPT-4, Gemini, Claude, Llama 3.3."
        )
    },
    {
        "intent"   : "groq",
        "keywords" : ["what is groq","groq","llama","llama 3"],
        "response" : (
            "Groq is a free AI inference platform. I use it as my L4 fallback. "
            "It runs Llama 3.3-70B at blazing speed. Free tier: 30 req/min. "
            "Get your key at console.groq.com"
        )
    },
    {
        "intent"   : "decodelabs",
        "keywords" : ["decodelabs","decode labs","this internship","this training"],
        "response" : (
            "DecodeLabs is an AI training program in Greater Lucknow, India. "
            "It teaches real-world AI engineering through hands-on projects. "
            "You're currently in Batch 2026!"
        )
    },
    {
        "intent"   : "joke",
        "keywords" : ["joke","tell me a joke","make me laugh","funny"],
        "response" : "Why do programmers prefer dark mode? Because light attracts bugs! 🐛"
    },
    {
        "intent"   : "motivation",
        "keywords" : ["motivate me","i am tired","i'm tired","feeling low",
                      "i give up","this is hard","encourage me","inspire me"],
        "response" : (
            "An LLM without rules is a hallucination engine. "
            "YOU are the rules. Every line of code you write today "
            "is a brick in the foundation of your career. Keep going. 💪"
        )
    },
    {
        "intent"   : "thanks",
        "keywords" : ["thank you","thanks","thx","ty","appreciate it","great job"],
        "response" : "Happy to help! That's what I'm here for. 😊"
    },
    {
        "intent"   : "projects",
        "keywords" : ["what are the projects","project 1","project 2","project 3",
                      "decodelabs projects"],
        "response" : (
            "DecodeLabs has 3 projects:\n"
            "  P1 → Rule-Based AI Chatbot (this one!)\n"
            "  P2 → Data Classification Using AI (KNN + Iris)\n"
            "  P3 → AI Recommendation Logic (TF-IDF + Cosine Similarity)"
        )
    },
]


# ═══════════════════════════════════════════════════════════════
#  LEVEL 3 — REGEX PATTERNS (Dynamic Question Handling)
# ═══════════════════════════════════════════════════════════════

def handle_what_is(match):
    topic = match.group(1).strip()
    for intent in INTENT_GROUPS:
        if any(topic in kw for kw in intent["keywords"]):
            return intent["response"]
    return None  # Escalate to Groq

def handle_how_to(match):
    return None  # Always escalate to Groq

def handle_tell_me_about(match):
    return None  # Escalate to Groq

def handle_difference(match):
    return None  # Escalate to Groq

def handle_time(_match):
    import datetime
    now = datetime.datetime.now()
    return f"Current time: {now.strftime('%I:%M %p, %d %B %Y')}"

REGEX_PATTERNS = [
    {
        "pattern" : re.compile(r"what is (.+)",            re.IGNORECASE),
        "handler" : handle_what_is
    },
    {
        "pattern" : re.compile(r"what are (.+)",           re.IGNORECASE),
        "handler" : handle_what_is
    },
    {
        "pattern" : re.compile(r"explain (.+)",            re.IGNORECASE),
        "handler" : handle_what_is
    },
    {
        "pattern" : re.compile(r"define (.+)",             re.IGNORECASE),
        "handler" : handle_what_is
    },
    {
        "pattern" : re.compile(r"how (?:do|can|to) (.+)", re.IGNORECASE),
        "handler" : handle_how_to
    },
    {
        "pattern" : re.compile(r"tell me about (.+)",      re.IGNORECASE),
        "handler" : handle_tell_me_about
    },
    {
        "pattern" : re.compile(r"difference between (.+) and (.+)", re.IGNORECASE),
        "handler" : handle_difference
    },
    {
        "pattern" : re.compile(r"(?:what(?:'s| is) the )?(?:current )?time", re.IGNORECASE),
        "handler" : handle_time
    },
]


# ═══════════════════════════════════════════════════════════════
#  INPUT SANITIZATION
# ═══════════════════════════════════════════════════════════════

def sanitize(raw_input: str) -> str:
    return raw_input.lower().strip()


# ═══════════════════════════════════════════════════════════════
#  CORE RESPONSE ENGINE — 4-LEVEL PIPELINE
# ═══════════════════════════════════════════════════════════════

def get_response(clean_input: str, groq_client, conversation_history: list) -> tuple:
    """
    4-level pipeline:
    L2 → Intent keyword matching
    L3 → Regex dynamic pattern matching
    L4 → Groq Llama 3.3 fallback

    Returns: (response_text, source_label)
    """

    # ── L2: KEYWORD SCANNING ─────────────────────────────────
    for intent in INTENT_GROUPS:
        for keyword in intent["keywords"]:
            if keyword in clean_input:
                return (intent["response"], f"[L2-Rule:{intent['intent']}]")

    # ── L3: REGEX PATTERN MATCHING ────────────────────────────
    for item in REGEX_PATTERNS:
        match = item["pattern"].search(clean_input)
        if match:
            result = item["handler"](match)
            if result:
                return (result, "[L3-Regex]")
            else:
                break  # Fall to Groq with original input

    # ── L4: GROQ LLAMA 3.3 FALLBACK ──────────────────────────
    if groq_client:
        response = ask_groq(groq_client, conversation_history, clean_input)
        return (response, "[L4-Groq-Llama3.3]")

    # ── HARD FALLBACK (no API key) ────────────────────────────
    return (
        "I don't know that yet. Add your Groq API key in chatbot_v2.py "
        "to unlock Llama 3.3 responses! Get free key at console.groq.com",
        "[Fallback]"
    )


# ═══════════════════════════════════════════════════════════════
#  EXIT COMMANDS
# ═══════════════════════════════════════════════════════════════

EXIT_COMMANDS = {"exit", "quit", "stop", "bye", "goodbye", "/exit", "/quit"}


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP — THE HEARTBEAT
# ═══════════════════════════════════════════════════════════════

def run_chatbot():
    groq_client = init_groq()

    conversation_history = []

    print("\n" + "=" * 58)
    print(f"  {BOT_NAME} v2 — Hybrid Chatbot | DecodeLabs Batch 2026")
    print(f"  Levels: L2-Intent | L3-Regex", end="")
    if groq_client:
        print(f" | L4-Groq Llama 3.3 ✓")
    else:
        print(f"\n  [!] Groq not configured — pure rule mode active")
        print(f"      Add key at console.groq.com → paste in GROQ_API_KEY")
    print("  Type 'help' for options. Type 'exit' to quit.")
    print("=" * 58 + "\n")

    while True:

        # ── INPUT ─────────────────────────────────────────────
        try:
            raw_input = input("You  : ")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{BOT_NAME}: Caught interrupt. Goodbye! 🚀")
            break

        clean_input = sanitize(raw_input)

        if not clean_input:
            print(f"{BOT_NAME}: Say something!\n")
            continue

        # ── EXIT CHECK ────────────────────────────────────────
        if clean_input in EXIT_COMMANDS:
            print(f"{BOT_NAME}: Goodbye! Keep building. 🚀\n")
            break

        # ── RESPONSE ──────────────────────────────────────────
        response, source = get_response(clean_input, groq_client, conversation_history)
        print(f"{BOT_NAME}: {response}")
        print(f"       ↳ {source}\n")

        # ── UPDATE HISTORY ────────────────────────────────────
        conversation_history.append({"role": "user",      "content": raw_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Keep last 20 turns
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    run_chatbot()
