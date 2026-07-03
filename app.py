"""Auto-detect to English translator over the free MyMemory API.
MyMemory rejects "auto" as a source language, so we detect locally first."""

import os
import re
import time
from collections import defaultdict, deque

import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect_langs, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # deterministic detection

app = Flask(__name__)

MYMEMORY_URL = "https://api.mymemory.translated.net/get"

# Set MYMEMORY_EMAIL to raise the daily quota (passed as MyMemory's "de" param).
CONTACT_EMAIL = os.environ.get("MYMEMORY_EMAIL", "").strip()

MAX_CHUNK_BYTES = 480  # MyMemory caps each request at 500 bytes of input

MAX_INPUT_BYTES = 5000  # cap one request's input so a giant paste can't hog the worker

# Rate limiting for /api/translate. In-memory and per-process, which is enough
# for one small box. A restart clears the counters and that's fine.
RATE_LIMIT_MAX = 10  # requests allowed per window per address
RATE_LIMIT_WINDOW = 60  # seconds
_rate_buckets = defaultdict(deque)  # address -> recent request times

# Short input detects poorly, so we let the translation correct the guess: if
# MyMemory echoes the input back, the source was wrong and we try the next of
# these, ordered by rough global commonness.
FALLBACK_LANGS = ["es", "fr", "de", "it", "pt", "nl", "ru", "pl", "tr", "sv", "id", "vi"]

MAX_DETECT_ATTEMPTS = 6  # bound the MyMemory round trips one short phrase can cost

LANG_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}


def byte_len(text):
    return len(text.encode("utf-8"))


def _now():
    """Split out so tests can move the clock without patching time itself."""
    return time.monotonic()


def rate_limited(address):
    """True when this address has spent its allowance for the current window.
    The address is best-effort identity. request.remote_addr is the direct
    peer, so behind a proxy every client can share one bucket unless the proxy
    forwards the real address. Good enough for a small self-hosted app."""
    bucket = _rate_buckets[address]
    now = _now()
    while bucket and now - bucket[0] >= RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX:
        return True
    bucket.append(now)
    return False


def chunk_text(text):
    text = text.strip()
    if not text:
        return []
    if byte_len(text) <= MAX_CHUNK_BYTES:
        return [text]

    pieces = [s for s in re.split(r"(?<=[.!?。！？\n])\s*", text) if s]
    chunks = []
    buf = ""
    for piece in pieces:
        if byte_len(piece) > MAX_CHUNK_BYTES:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(hard_split(piece))
            continue
        candidate = (buf + " " + piece).strip() if buf else piece
        if byte_len(candidate) <= MAX_CHUNK_BYTES:
            buf = candidate
        else:
            chunks.append(buf)
            buf = piece
    if buf:
        chunks.append(buf)
    return chunks


def hard_split(piece):
    out = []
    buf = ""
    for word in piece.split(" "):
        candidate = (buf + " " + word).strip() if buf else word
        if byte_len(candidate) <= MAX_CHUNK_BYTES:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            if byte_len(word) <= MAX_CHUNK_BYTES:
                buf = word
            else:
                out.extend(split_by_chars(word))
                buf = ""
    if buf:
        out.append(buf)
    return out


def split_by_chars(token):
    out, buf = [], ""
    for ch in token:
        if byte_len(buf + ch) > MAX_CHUNK_BYTES:
            out.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def translate_chunk(chunk, source_lang):
    params = {"q": chunk, "langpair": f"{source_lang}|en"}
    if CONTACT_EMAIL:
        params["de"] = CONTACT_EMAIL

    resp = requests.get(MYMEMORY_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("responseStatus") not in (200, "200"):  # quota / service error
        raise RuntimeError(data.get("responseDetails") or "Translation service error")
    body = data.get("responseData")
    if not isinstance(body, dict):
        # MyMemory sometimes puts an error string here even with status 200.
        raise RuntimeError(f"Unexpected responseData shape: {body!r}")
    return body.get("translatedText", "")


def normalize(text):
    return " ".join(text.split()).casefold()


def is_mirror(source, translated):
    """MyMemory echoes the input back when it can't translate from the source
    language we asked for. That echo is our signal the language guess was wrong."""
    translated = translated.strip()
    return not translated or normalize(translated) == normalize(source)


def candidate_langs(ranked):
    """Source languages to try, best first. Detection's ranking comes first, then
    the common-language fallback, with English and duplicates removed."""
    seen = set()
    result = []
    for code in [*ranked, *FALLBACK_LANGS]:
        if code != "en" and code not in seen:
            seen.add(code)
            result.append(code)
    return result


def translate_short(chunk, ranked):
    """Translate a short chunk, trying candidates until one returns more than the
    input echoed back. Returns (source_lang, translation), or (None, None)."""
    for source_lang in candidate_langs(ranked)[:MAX_DETECT_ATTEMPTS]:
        translated = translate_chunk(chunk, source_lang)
        if not is_mirror(chunk, translated):
            return source_lang, translated
    return None, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    # CORS-open health check so external monitors can read the status cross-origin.
    resp = jsonify(status="ok")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/api/translate", methods=["POST"])
def api_translate():
    if rate_limited(request.remote_addr or "unknown"):
        return jsonify({"error": "Slow down a little. Try again in a minute."}), 429

    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Type something to translate."}), 400
    if byte_len(text) > MAX_INPUT_BYTES:
        return jsonify({"error": "That's too much text at once. Keep it under 5,000 characters."}), 413

    try:
        ranked = [guess.lang for guess in detect_langs(text)]
    except LangDetectException:
        ranked = []

    if ranked[:1] == ["en"]:
        return jsonify({
            "detected_code": "en",
            "detected_name": "English",
            "already_english": True,
            "translation": text,
        })

    chunks = chunk_text(text)
    try:
        if len(chunks) == 1:
            source_lang, translated = translate_short(chunks[0], ranked)
        elif ranked:
            source_lang = ranked[0]
            translated = " ".join(translate_chunk(c, source_lang) for c in chunks).strip()
        else:
            source_lang = None
        if source_lang is None:
            return jsonify({"error": "Couldn't figure out the language. Try a longer phrase."}), 422
    except RuntimeError as e:
        app.logger.warning("MyMemory returned an error: %s", e)
        return jsonify({"error": "The translation service returned an error. Try again in a minute."}), 502
    except requests.RequestException as e:
        app.logger.warning("Could not reach MyMemory: %r", e)
        return jsonify({"error": "Couldn't reach the translation service. Check your connection."}), 502

    return jsonify({
        "detected_code": source_lang,
        "detected_name": LANG_NAMES.get(source_lang, source_lang),
        "already_english": False,
        "translation": translated,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)  # 0.0.0.0 so Render and the LAN can reach it
