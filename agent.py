"""
agent.py — Core AI Calling Agent Logic
Handles: intent detection, KB lookup, LLM fallback, session memory, call summary
"""

import json
import os
import re
from datetime import datetime

# ── Load Knowledge Base ───────────────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)

# ── LLM Setup (Groq) ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

_llm_client = None
_use_groq = False

if GROQ_API_KEY:
    try:
        from groq import Groq
        _llm_client = Groq(api_key=GROQ_API_KEY)
        _use_groq = True
        print("[Agent] LLM: Groq llama3-8b-8192 ready")
    except ImportError:
        print("[Agent] groq package missing — run: pip install groq")

# ── Session Store ─────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],               # [{role, content}, ...]
            "turn_count": 0,
            "intents_seen": [],          # ordered list for summary
            "awaiting_order_id": False,  # multi-turn order lookup state
            "started_at": datetime.now().strftime("%H:%M:%S"),
        }
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def get_summary(session_id: str) -> str:
    """Return a human-readable call summary."""
    session = _sessions.get(session_id)
    if not session or session["turn_count"] == 0:
        return "No conversation recorded for this session."

    unique_intents = list(dict.fromkeys(session["intents_seen"]))
    topics = ", ".join(unique_intents) if unique_intents else "general inquiry"
    turns = session["turn_count"]
    started = session.get("started_at", "N/A")

    return (
        f"📋 Call Summary | Started: {started} | "
        f"Turns: {turns} | Topics: {topics}."
    )


# ── LLM Inference ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a polite and helpful AI customer support agent for an e-commerce company. "
    "Keep answers concise (2-3 sentences max). "
    "If you are unsure, say so honestly and offer to connect the customer to a human agent. "
    "Do not make up order details or policies."
)


def _call_llm(user_input: str, history: list) -> str:
    messages = (
        [{"role": "system", "content": _SYSTEM_PROMPT}]
        + history[-6:]
        + [{"role": "user", "content": user_input}]
    )

    if _use_groq and _llm_client:
        resp = _llm_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=150,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()

    return ""


# ── Intent Detection ──────────────────────────────────────────────────────────
_INTENT_MAP: dict[str, list[str]] = {
    "greeting":     ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy"],
    "closing":      ["bye", "goodbye", "see you", "thank you", "thanks", "that's all", "all done", "done", "exit", "quit"],
    "order_status": ["order", "track", "tracking", "status", "delivery", "where is my", "shipped", "dispatch", "parcel", "package"],
    "refund":       ["refund", "money back", "get my money", "reimburs"],
    "return":       ["return", "send back", "exchange", "swap"],
    "cancel":       ["cancel", "cancellation", "stop my order"],
    "shipping":     ["shipping", "delivery time", "how long", "how many days", "when will"],
    "payment":      ["payment", "pay", "credit card", "debit card", "upi", "net banking", "cash on delivery", "cod"],
    "contact":      ["contact", "email", "phone number", "reach", "call you", "support number"],
    "hours":        ["hours", "timing", "open", "available", "working hours", "office hours"],
    "warranty":     ["warranty", "guarantee", "defective", "broken", "damaged"],
    "discount":     ["discount", "coupon", "promo", "offer", "deal", "voucher"],
    "complaint":    ["complaint", "not happy", "unhappy", "issue", "problem", "bad experience", "worst", "terrible", "angry", "frustrated", "disappointed"],
    "human":        ["human", "agent", "representative", "real person", "speak to someone", "talk to someone", "live agent"],
    "product":      ["product", "item", "catalog", "available", "stock", "price", "cost", "how much"],
}


def detect_intent(text: str) -> str:
    t = text.lower()
    for intent, keywords in _INTENT_MAP.items():
        if any(kw in t for kw in keywords):
            return intent
    return "general"


# ── Order ID Extraction ───────────────────────────────────────────────────────
def _extract_order_id(text: str) -> str | None:
    """Extract a 4-6 digit order ID from text."""
    match = re.search(r"\b(\d{4,6})\b", text)
    return match.group(1) if match else None


# ── Main Response Function ────────────────────────────────────────────────────
def get_response(user_input: str, session_id: str = "default") -> dict:
    """
    Process user input and return agent response.

    Returns:
        dict with keys: response, intent, escalate, turn
    """
    user_input = user_input.strip()
    if not user_input:
        return {
            "response": KB["policies"]["fallback"],
            "intent": "empty",
            "escalate": False,
            "turn": 0,
        }

    session = _get_session(session_id)
    history = session["history"]
    intent = detect_intent(user_input)
    escalate = False
    response = ""

    # ── Greeting ──────────────────────────────────────────────────────────────
    if intent == "greeting":
        response = (
            KB["policies"]["greeting"]
            if session["turn_count"] == 0
            else "Hello again! What else can I help you with?"
        )

    # ── Closing / End of call ─────────────────────────────────────────────────
    elif intent == "closing":
        summary = get_summary(session_id)
        response = KB["policies"]["closing"] + " " + summary
        # Update history before clearing so summary is accurate
        session["history"].append({"role": "user", "content": user_input})
        session["history"].append({"role": "assistant", "content": response})
        session["turn_count"] += 1
        session["intents_seen"].append(intent)
        clear_session(session_id)
        return {"response": response, "intent": intent, "escalate": False, "turn": session["turn_count"]}

    # ── Escalation: human agent request ──────────────────────────────────────
    elif intent == "human":
        response = KB["policies"]["escalation_message"]
        escalate = True

    # ── Complaint → escalate ──────────────────────────────────────────────────
    elif intent == "complaint":
        response = (
            "I'm really sorry to hear that you're having a bad experience. "
            + KB["policies"]["escalation_message"]
        )
        escalate = True

    # ── Order Status (multi-turn aware) ───────────────────────────────────────
    elif intent == "order_status" or session["awaiting_order_id"]:
        order_id = _extract_order_id(user_input)

        if order_id and order_id in KB["orders"]:
            info = KB["orders"][order_id]
            response = (
                f"Your order #{order_id} is currently '{info['status']}'. "
                f"Estimated arrival: {info['eta']}."
            )
            session["awaiting_order_id"] = False
        elif order_id:
            response = (
                f"I couldn't find order #{order_id} in our system. "
                "Please double-check the order ID or contact support."
            )
            session["awaiting_order_id"] = False
        else:
            # No order ID provided yet — ask for it
            response = "Sure! Could you please share your order ID so I can look that up for you?"
            session["awaiting_order_id"] = True
            intent = "order_status"

    # ── Direct FAQ intent match ───────────────────────────────────────────────
    elif intent in KB["faqs"]:
        response = KB["faqs"][intent]

    # ── Partial FAQ keyword match ─────────────────────────────────────────────
    else:
        for key, answer in KB["faqs"].items():
            if key in user_input.lower():
                response = answer
                intent = "faq"
                break

    # ── LLM fallback for anything unhandled ───────────────────────────────────
    if not response:
        try:
            response = _call_llm(user_input, history)
            if response:
                intent = "llm_response"
        except Exception as exc:
            print(f"[Agent] LLM error: {exc}")

    # ── Final fallback ────────────────────────────────────────────────────────
    if not response:
        response = KB["policies"]["fallback"]

    # ── Persist to session ────────────────────────────────────────────────────
    session["history"].append({"role": "user", "content": user_input})
    session["history"].append({"role": "assistant", "content": response})
    session["turn_count"] += 1
    if intent != "general":
        session["intents_seen"].append(intent)

    return {
        "response": response,
        "intent": intent,
        "escalate": escalate,
        "turn": session["turn_count"],
    }
