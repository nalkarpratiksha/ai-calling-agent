"""
main.py — FastAPI server for AI Calling Agent
Endpoints: /chat, /voice/stt, /voice/tts, /voice/call, /session/summary, /session/reset
"""

import os
import base64
import tempfile
import subprocess
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import get_response, clear_session, get_summary

# ── Voice: Whisper STT ────────────────────────────────────────────────────────
try:
    import whisper
    WHISPER_MODEL = whisper.load_model("tiny")
    WHISPER_AVAILABLE = True
    print("[Main] Whisper STT: ready")
except Exception as e:
    WHISPER_AVAILABLE = False
    print(f"[Main] Whisper unavailable: {e}")

# ── Voice: gTTS ───────────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    print("[Main] gTTS TTS: ready")
except Exception as e:
    TTS_AVAILABLE = False
    print(f"[Main] gTTS unavailable: {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Calling Agent",
    description=(
        "AI-powered customer support agent. "
        "Supports text chat, speech-to-text, text-to-speech, and full voice call pipeline."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    intent: str
    escalate: bool
    turn: int
    session_id: str

class TTSRequest(BaseModel):
    text: str

class SessionRequest(BaseModel):
    session_id: str

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_upload(data: bytes, suffix: str) -> str:
    """Write bytes to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
    finally:
        tmp.close()  # Must close before ffmpeg can read it on Windows
    return tmp.name


def _find_ffmpeg() -> str:
    """Find ffmpeg binary — checks PATH and common Windows install locations."""
    # Inject known ffmpeg bin dir into PATH so subprocess can find it
    ffmpeg_bin = r"C:\ffmpeg-8.1-essentials_build\bin"
    if ffmpeg_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

    found = shutil.which("ffmpeg")
    if found:
        return found
    # Common Windows locations to try
    candidates = [
        r"C:\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise RuntimeError(
        "ffmpeg not found. Make sure C:\\ffmpeg-8.1-essentials_build\\bin is in your PATH "
        "and restart the server."
    )


def _convert_to_wav(src_path: str) -> str:
    """Convert any audio format to wav using ffmpeg. Returns new wav path."""
    wav_path = src_path.rsplit(".", 1)[0] + "_converted.wav"
    ffmpeg = _find_ffmpeg()
    print(f"[Voice] Using ffmpeg at: {ffmpeg}")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True, text=True,
        shell=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    if not os.path.exists(wav_path):
        raise RuntimeError(f"ffmpeg ran but output file not created: {wav_path}")
    return wav_path

def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _tts_to_b64(text: str) -> str:
    path = tempfile.mktemp(suffix=".mp3")
    try:
        gTTS(text=text, lang="en").save(path)
        return _file_to_b64(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(content="", status_code=204)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Serve the web UI."""
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    with open(ui_path, encoding="utf-8") as f:
        return f.read()


@app.get("/health", tags=["System"])
def health():
    """Check service and feature availability."""
    return {
        "status": "ok",
        "whisper_stt": WHISPER_AVAILABLE,
        "gtts_tts": TTS_AVAILABLE,
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(req: ChatRequest):
    """
    Send a text message to the AI agent and receive a response.

    - Maintains conversation context per session_id
    - Detects intent and looks up knowledge base
    - Falls back to LLM for unknown queries
    """
    if not req.text.strip():
        raise HTTPException(400, "Message text cannot be empty.")

    result = get_response(req.text, req.session_id)
    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        escalate=result["escalate"],
        turn=result["turn"],
        session_id=req.session_id,
    )


@app.post("/voice/stt", tags=["Voice"])
async def stt(file: UploadFile = File(...)):
    """
    Convert an uploaded audio file to text using Whisper.

    Accepts: .wav, .mp3, .webm, .ogg, .m4a
    """
    if not WHISPER_AVAILABLE:
        raise HTTPException(503, "Whisper not installed. Run: pip install openai-whisper")

    suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
    path = _save_upload(await file.read(), suffix)
    wav_path = None

    try:
        # Convert to wav if not already wav (webm from browser needs ffmpeg)
        if suffix.lower() not in (".wav",):
            try:
                wav_path = _convert_to_wav(path)
                transcribe_path = wav_path
            except RuntimeError as e:
                raise HTTPException(503, str(e))
        else:
            transcribe_path = path

        result = WHISPER_MODEL.transcribe(transcribe_path, language="en", fp16=False)
        text = result["text"].strip()
    finally:
        if os.path.exists(path):
            os.unlink(path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

    return {"text": text}


@app.post("/voice/tts", tags=["Voice"])
def tts(req: TTSRequest):
    """
    Convert text to speech and return base64-encoded MP3 audio.
    """
    if not TTS_AVAILABLE:
        raise HTTPException(503, "gTTS not installed. Run: pip install gtts")
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")

    audio_b64 = _tts_to_b64(req.text)
    return {"audio_base64": audio_b64, "format": "mp3"}


@app.post("/voice/call", tags=["Voice"])
async def voice_call(
    file: UploadFile = File(...),
    session_id: str = Query(default="default", description="Session identifier"),
):
    """
    Full voice call pipeline: audio → STT → agent → TTS → audio response.

    Returns transcribed input, agent text response, and base64 MP3 audio.
    """
    if not WHISPER_AVAILABLE:
        raise HTTPException(503, "Whisper not installed.")
    if not TTS_AVAILABLE:
        raise HTTPException(503, "gTTS not installed.")

    # Step 1: STT
    suffix = os.path.splitext(file.filename or "audio.webm")[-1] or ".webm"
    stt_path = _save_upload(await file.read(), suffix)
    wav_path = None

    try:
        if suffix.lower() not in (".wav",):
            try:
                wav_path = _convert_to_wav(stt_path)
                transcribe_path = wav_path
            except RuntimeError as e:
                print(f"[Voice] Conversion error: {e}")
                raise HTTPException(503, str(e))
        else:
            transcribe_path = stt_path

        print(f"[Voice] Transcribing: {transcribe_path}")
        result = WHISPER_MODEL.transcribe(transcribe_path, language="en", fp16=False)
        user_text = result["text"].strip()
        print(f"[Voice] Transcribed: '{user_text}'")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Voice] STT error: {e}")
        raise HTTPException(500, f"STT failed: {str(e)}")
    finally:
        if os.path.exists(stt_path):
            os.unlink(stt_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

    if not user_text:
        raise HTTPException(422, "Could not transcribe audio. Please speak clearly and try again.")

    # Step 2: Agent
    result = get_response(user_text, session_id)

    # Step 3: TTS
    audio_b64 = _tts_to_b64(result["response"])

    return {
        "input_text": user_text,
        "response": result["response"],
        "intent": result["intent"],
        "escalate": result["escalate"],
        "turn": result["turn"],
        "audio_base64": audio_b64,
        "format": "mp3",
    }


@app.get("/session/summary", tags=["Session"])
def summary(session_id: str = Query(default="default")):
    """Get a summary of the current call session."""
    return {"summary": get_summary(session_id), "session_id": session_id}


@app.post("/session/reset", tags=["Session"])
def reset(req: SessionRequest):
    """Clear conversation history for a session."""
    clear_session(req.session_id)
    return {"message": f"Session '{req.session_id}' has been reset."}
