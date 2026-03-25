# AI Calling Agent — Customer Support

An AI-powered customer support calling agent built with FastAPI, Whisper STT, gTTS, and OpenAI/TinyLlama LLM.

---

## Features

- Text chat with multi-turn conversation memory
- Voice input (Whisper STT) + voice output (gTTS TTS)
- Full voice call pipeline: audio → agent → audio
- Intent detection: order status, refund, return, cancel, shipping, complaint, escalation, and more
- Knowledge base: orders, FAQs, policies
- Multi-turn order lookup (agent asks for order ID if not provided)
- Escalation to human agent on complaint or request
- Call summary at end of session
- Web UI with typing indicator, intent tags, and audio playback
- Swagger API docs at `/docs`

---

## Setup

```bash
cd ai-calling-agent

# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set OpenAI key for best LLM quality
#    Without this, the agent uses local TinyLlama or KB-only fallback
set OPENAI_API_KEY=sk-your-key-here    # Windows CMD
# export OPENAI_API_KEY=sk-...         # Mac / Linux

# 4. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Web UI:   http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## LLM Modes (auto-detected)

| Mode | How to enable | Notes |
|------|--------------|-------|
| OpenAI GPT-3.5 | Set `OPENAI_API_KEY` env var | Best quality, requires internet |
| TinyLlama (local) | Uncomment torch lines in requirements.txt | ~1GB RAM, offline |
| KB-only fallback | No setup needed | Works without any LLM |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/`                  | Web UI |
| GET  | `/health`            | Service + feature status |
| POST | `/chat`              | Text chat with agent |
| POST | `/voice/stt`         | Audio file → transcribed text |
| POST | `/voice/tts`         | Text → base64 MP3 audio |
| POST | `/voice/call`        | Full pipeline: audio → agent → audio |
| GET  | `/session/summary`   | Get call summary |
| POST | `/session/reset`     | Clear session history |

---

## Example Requests

**Order status (multi-turn)**
```
User:  "Where is my order?"
Agent: "Could you please share your order ID?"
User:  "12345"
Agent: "Your order #12345 is Out for delivery. Estimated arrival: Today by 6 PM."
```

**Refund**
```json
POST /chat
{ "text": "I want a refund", "session_id": "demo" }
```

**Escalation**
```json
POST /chat
{ "text": "I want to speak to a human agent", "session_id": "demo" }
```

**Call Summary**
```
GET /session/summary?session_id=demo
```

---

## System Architecture

```
User Input (Text / Audio)
        │
        ▼
  [STT] Whisper tiny          ← audio → text
        │
        ▼
  [Intent Detection]          ← keyword classifier
        │
        ├── KB Match           ← orders, FAQs, policies
        │
        └── LLM Fallback       ← OpenAI / TinyLlama
        │
        ▼
  [TTS] gTTS                  ← text → audio
        │
        ▼
  Response to User
```

---

## Test Orders

| Order ID | Status | ETA |
|----------|--------|-----|
| 12345 | Out for delivery | Today by 6 PM |
| 67890 | Processing | 2-3 business days |
| 11111 | Delivered | March 20 |
| 99999 | Cancelled | Refund in 3-5 days |
| 55555 | Shipped | Tomorrow by 8 PM |
| 33333 | Pending payment | Awaiting confirmation |

---

## Supported Intents

`greeting` · `order_status` · `refund` · `return` · `cancel` · `shipping` · `payment` · `contact` · `hours` · `warranty` · `discount` · `complaint` · `human` · `closing`
