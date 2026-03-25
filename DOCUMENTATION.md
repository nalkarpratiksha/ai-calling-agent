# AI Calling Agent — Project Documentation

## Overview

An AI-powered customer support calling agent that handles text and voice interactions. It understands customer queries, detects intent, looks up a knowledge base, and falls back to an LLM for unknown questions.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Backend API | FastAPI (Python) |
| Speech-to-Text | OpenAI Whisper (tiny) |
| Text-to-Speech | gTTS (Google TTS) |
| LLM | OpenAI GPT-3.5 / TinyLlama (local) |
| Knowledge Base | JSON file |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
ai-calling-agent/
├── agent.py            # Core logic: intent detection, KB lookup, LLM, session
├── main.py             # FastAPI server and all API endpoints
├── ui.html             # Browser-based chat + voice UI
├── knowledge_base.json # Orders, FAQs, and policies
├── requirements.txt    # Python dependencies
└── DOCUMENTATION.md    # This file
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set OpenAI key for better LLM responses
set OPENAI_API_KEY=sk-your-key-here   # Windows
export OPENAI_API_KEY=sk-...          # Mac/Linux

# 3. Start the server
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## System Architecture

```
User (Text / Voice)
       │
       ▼
  FastAPI Server (main.py)
       │
       ├─ /voice/stt ──► Whisper STT ──► text
       │
       ├─ /chat ────────► agent.py
       │                     │
       │               Intent Detection
       │                     │
       │            ┌────────┴────────┐
       │         KB Lookup        LLM Fallback
       │         (JSON)        (OpenAI / TinyLlama)
       │                     │
       └─ /voice/tts ◄────── response text
              │
              ▼
         gTTS → MP3 audio → User
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Server + feature status |
| POST | `/chat` | Text chat with agent |
| POST | `/voice/stt` | Audio file → text |
| POST | `/voice/tts` | Text → MP3 audio (base64) |
| POST | `/voice/call` | Full pipeline: audio → agent → audio |
| GET | `/session/summary` | Get call summary |
| POST | `/session/reset` | Clear session |

---

## Core Features

**Intent Detection** — keyword-based classifier covering:
`greeting` · `order_status` · `refund` · `return` · `cancel` · `shipping` · `payment` · `contact` · `hours` · `warranty` · `discount` · `complaint` · `human` · `closing`

**Multi-turn Conversation** — session memory tracks history, turn count, and intents seen per `session_id`.

**Order Lookup** — if user says "where is my order" without an ID, agent asks for it (multi-turn state).

**Escalation** — complaint or "speak to human" triggers escalation flag and escalation message.

**Call Summary** — generated on session close or on demand via `/session/summary`.

**LLM Fallback** — anything not matched by KB goes to OpenAI GPT-3.5 (or local TinyLlama if no API key).

---

## Sample Conversation

```
Agent: Hello! Thank you for calling customer support. How can I help you?
User:  Where is my order?
Agent: Could you please share your order ID?
User:  12345
Agent: Your order #12345 is Out for delivery. Estimated arrival: Today by 6 PM.
User:  Thanks, bye!
Agent: Thank you for contacting us. Have a great day!
       📋 Call Summary | Turns: 3 | Topics: order_status, closing.
```

---

## Knowledge Base (knowledge_base.json)

- **orders** — order ID → status + ETA (test IDs: 12345, 67890, 11111, 99999, 55555, 33333)
- **faqs** — refund, return, shipping, payment, contact, hours, warranty, discount, etc.
- **policies** — greeting, fallback, escalation message, closing message

---

## LLM Modes

| Mode | Requirement | Quality |
|------|-------------|---------|
| OpenAI GPT-3.5 | `OPENAI_API_KEY` env var | Best |
| TinyLlama (local) | Uncomment torch lines in requirements.txt | Good |
| KB-only fallback | Nothing | Basic |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `0.0.0.0:8000` not opening | Use `http://localhost:8000` |
| Whisper not found | `pip install openai-whisper` |
| gTTS not found | `pip install gtts` |
| Server not starting | Make sure venv is activated and you're in the project folder |
| Microphone not working | Allow mic permissions in browser |
