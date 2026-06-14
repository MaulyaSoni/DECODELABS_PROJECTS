# ============================================================
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  Project 1  : Rule-Based AI Chatbot — VERSION 1
#  Type       : Pure Rule-Based | Zero External Dependencies
#  Engine     : Dictionary O(1) Lookup + Infinite While Loop
#  Spec       : Exactly as per DecodeLabs Project 1 blueprint
# ============================================================
#
#  HOW TO RUN:
#  python chatbot_v1.py
#
#  NO pip install needed — uses only Python built-ins
# ============================================================

# ── KNOWLEDGE BASE (Dictionary / Hash-Map) ───────────────────
# Key   = sanitized user intent (lowercase, stripped)
# Value = bot's hard-coded deterministic response
# O(1) lookup — avoids the if-elif ladder anti-pattern

KNOWLEDGE_BASE = {
    # Greetings
    "hello"           : "Hey there! I'm DBot, your AI assistant. How can I help?",
    "hi"              : "Hi! Great to see you. What's on your mind?",
    "hey"             : "Hey! I'm listening. Ask me anything.",
    "good morning"    : "Good morning! Ready to build something great today?",
    "good evening"    : "Good evening! Hope your day went well.",
    "good afternoon"  : "Good afternoon! How can I assist you?",

    # Farewells (also handled as exit — see EXIT_COMMANDS set)
    "bye"             : "Goodbye! Keep building. 🚀",
    "goodbye"         : "See you on the other side of the terminal!",

    # Identity
    "who are you"     : "I'm DBot — a rule-based AI chatbot built at DecodeLabs.",
    "what are you"    : "A deterministic logic engine. No hallucinations. 100% hard-coded.",
    "your name"       : "My name is DBot. Short for DecodeLabs Bot.",
    "are you a bot"   : "Yes! A proud rule-based bot. No neural network, just pure logic.",
    "are you human"   : "Nope! I'm DBot — a programmatic decision-making engine.",

    # Capability
    "what can you do" : "I can answer predefined questions using rule-based logic. Type 'help' to see options.",
    "help"            : (
        "Try asking me:\n"
        "  → 'hello' / 'hi' / 'hey'\n"
        "  → 'who are you' / 'what can you do'\n"
        "  → 'what is ai' / 'what is ml' / 'what is llm'\n"
        "  → 'tell me a joke' / 'motivate me'\n"
        "  → 'what time is it'\n"
        "  → 'bye' / 'exit' to quit"
    ),

    # Creator
    "who made you"    : "I was crafted by an AI intern at DecodeLabs, Batch 2026.",
    "who created you" : "A future AI Engineer following the DecodeLabs training kit!",
    "who built you"   : "Built by a DecodeLabs intern as Project 1 of the AI training kit.",

    # AI Concepts
    "what is ai"      : (
        "AI (Artificial Intelligence) is the simulation of human intelligence "
        "by machines — enabling them to learn, reason, and make decisions."
    ),
    "what is ml"      : (
        "ML (Machine Learning) is a subset of AI where systems automatically "
        "learn patterns from data without being explicitly programmed for each task."
    ),
    "what is llm"     : (
        "LLMs (Large Language Models) are massive neural networks trained on text. "
        "They predict the next token to generate human-like responses. "
        "Examples: GPT-4, Gemini, Claude, Llama."
    ),
    "what is deep learning" : (
        "Deep Learning uses multi-layered neural networks to learn complex patterns. "
        "CNNs handle images, RNNs handle sequences, Transformers power modern LLMs."
    ),
    "what is nlp"     : (
        "NLP (Natural Language Processing) is a branch of AI that helps machines "
        "understand, interpret, and generate human language."
    ),

    # DecodeLabs
    "what is decodelabs" : (
        "DecodeLabs is an AI training program based in Greater Lucknow, India. "
        "It teaches real-world AI engineering through hands-on projects."
    ),

    # Fun
    "tell me a joke"  : "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
    "joke"            : "I told a joke once. It had O(1) punchline delivery. Very efficient.",
    "another joke"    : "Why did the AI break up with the rule engine? Too many conditions!",

    # Motivation
    "motivate me"     : (
        "An LLM without rules is a hallucination engine.\n"
        "YOU are the rules. Every line of code you write today\n"
        "is a brick in the foundation of your career. Keep going! 💪"
    ),
    "i am tired"      : "Rest is a feature, not a bug. Come back stronger.",
    "i give up"       : "Bugs are just undiscovered features. Debug yourself and keep going!",

    # Time
    "what time is it" : "TIME_DYNAMIC",   # handled specially below
    "time"            : "TIME_DYNAMIC",

    # Thanks
    "thank you"       : "Happy to help! That's what I'm here for. 😊",
    "thanks"          : "You're welcome! Keep coding!",

    # About the project
    "what is project 1"   : (
        "Project 1 is the Rule-Based AI Chatbot — the foundation phase "
        "of the DecodeLabs AI training kit. It teaches control flow, "
        "dictionary lookup, and deterministic guardrails."
    ),
    "what are the projects": (
        "DecodeLabs has 3 projects:\n"
        "  P1 → Rule-Based AI Chatbot\n"
        "  P2 → Data Classification Using AI (KNN)\n"
        "  P3 → AI Recommendation Logic (TF-IDF + Cosine Similarity)"
    ),
}

# ── EXIT COMMANDS (Set for O(1) membership check) ────────────
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "stop", "/exit", "/quit"}

BOT_NAME = "DBot"


# ── INPUT SANITIZATION ────────────────────────────────────────
# Phase 1 of the IPO model
# .lower() → removes case sensitivity
# .strip() → removes leading/trailing whitespace

def sanitize(raw_input: str) -> str:
    return raw_input.lower().strip()


# ── RESPONSE ENGINE (.get() Method) ──────────────────────────
# Single atomic operation: lookup + fallback in one line
# O(1) time complexity — no cascading if-elif failures

def get_response(clean_input: str) -> str:
    import datetime

    # Special dynamic responses
    response = KNOWLEDGE_BASE.get(clean_input)

    if response == "TIME_DYNAMIC":
        now = datetime.datetime.now()
        return f"Current time: {now.strftime('%I:%M %p, %d %B %Y')}"

    if response:
        return response

    # Fallback
    return (
        "I don't understand that yet. "
        "Type 'help' to see what I can do, "
        "or upgrade to chatbot_v2.py for AI-powered responses!"
    )


# ── MAIN LOOP (The Heartbeat — Infinite Loop) ─────────────────
# while True keeps the organism alive
# EXIT_COMMANDS check = Kill Command → break

def run_chatbot():
    print("\n" + "=" * 55)
    print(f"  {BOT_NAME} v1 — Rule-Based Chatbot | DecodeLabs 2026")
    print("  Pure rule-based. No internet required.")
    print("  Type 'help' for options. Type 'exit' to quit.")
    print("=" * 55 + "\n")

    while True:

        # ── INPUT PHASE ──────────────────────────────────────
        try:
            raw_input = input("You  : ")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{BOT_NAME}: Caught interrupt. Goodbye! 🚀")
            break

        clean_input = sanitize(raw_input)

        # Guard: empty input
        if not clean_input:
            print(f"{BOT_NAME}: Say something! I'm all ears.\n")
            continue

        # ── EXIT CHECK (Kill Command) ─────────────────────────
        if clean_input in EXIT_COMMANDS:
            print(f"{BOT_NAME}: Goodbye! Keep building. 🚀\n")
            break

        # ── PROCESS + OUTPUT ──────────────────────────────────
        response = get_response(clean_input)
        print(f"{BOT_NAME}: {response}\n")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    run_chatbot()
