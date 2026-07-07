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

# MyMemory rejects long queries with "MAX ALLOWED QUERY : 500 CHARS". Packing
# by UTF-8 bytes stays under that cap because every char is at least one byte.
MAX_CHUNK_BYTES = 480

MAX_INPUT_BYTES = 5000  # cap one request's input so a giant paste can't hog the worker

# One session so MyMemory calls reuse a pooled connection instead of paying a
# fresh TLS handshake per request.
_http = requests.Session()

# langdetect emits lowercase region codes like zh-cn, but MyMemory's langpair
# wants the RFC3066 casing. Map the region-coded languages before building it.
MYMEMORY_LANG_CODES = {"zh-cn": "zh-CN", "zh-tw": "zh-TW"}

# Rate limiting for /api/translate. In-memory and per-process, which is enough
# for one small box. A restart clears the counters and that's fine.
RATE_LIMIT_MAX = 10  # requests allowed per window per address
RATE_LIMIT_WINDOW = 60  # seconds
RATE_PRUNE_EVERY = 100  # rate-limit calls between sweeps of idle buckets
_rate_buckets = defaultdict(deque)  # address -> recent request times
_rate_calls = 0  # counts calls so the sweep runs every RATE_PRUNE_EVERY

# The rate limiter keys on the direct peer address by default, which can't be
# spoofed. Behind a reverse proxy every visitor shares the proxy's address, so
# set TRUST_PROXY=1 to key on the rightmost X-Forwarded-For hop instead. The
# trusted proxy appends the true client address last, while the leftmost hops
# are client supplied and forgeable. Only enable it when a proxy you control
# sets that header.
TRUST_PROXY = os.environ.get("TRUST_PROXY", "").strip() == "1"

# Short input detects poorly, so we let the translation correct the guess. If
# MyMemory echoes the input back, the source was wrong and we try the next of
# these, ordered by rough global commonness.
FALLBACK_LANGS = ["es", "fr", "de", "it", "pt", "nl", "ru", "pl", "tr", "sv", "id", "vi"]

MAX_DETECT_ATTEMPTS = 6  # bound the MyMemory round trips one short phrase can cost

# A mirrored echo is usually a wrong language guess, but cognates like "no" or
# "Hotel" really do translate to themselves. When the input is one short word
# and langdetect put at least this much probability on the language we tried,
# we trust the echo as a real translation. 0.7 keeps weak guesses retrying
# while letting confident single-word detections through.
MIRROR_CONFIDENCE = 0.7

MAX_COGNATE_CHARS = 12  # one word under this length can pass as a cognate

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


def client_address():
    """Best-effort client identity for the rate limiter. The direct peer by
    default, or the rightmost X-Forwarded-For hop when TRUST_PROXY is on.
    The one trusted proxy appends the true client last, so the rightmost
    entry is reliable while the leftmost is client controlled."""
    if TRUST_PROXY:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return request.remote_addr or "unknown"


def _prune_idle_buckets(now):
    """Drop buckets whose newest entry has aged out of the window, so idle
    addresses don't accumulate in _rate_buckets forever."""
    # Snapshot the items so a concurrent insert from another worker thread
    # can't change the dict size mid iteration.
    idle = [addr for addr, bucket in list(_rate_buckets.items())
            if not bucket or now - bucket[-1] >= RATE_LIMIT_WINDOW]
    for addr in idle:
        del _rate_buckets[addr]


def rate_limited(address):
    """True when this address has spent its allowance for the current window."""
    global _rate_calls
    now = _now()
    _rate_calls += 1
    if _rate_calls % RATE_PRUNE_EVERY == 0:
        _prune_idle_buckets(now)
    bucket = _rate_buckets[address]
    while bucket and now - bucket[0] >= RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX:
        return True
    bucket.append(now)
    return False


def pack(pieces, sep, split_oversize=None):
    """Greedy byte-cap packer shared by the three splitters. Joins pieces with
    sep while each chunk stays within MAX_CHUNK_BYTES. A piece that alone
    exceeds the cap goes through split_oversize when one is given, so every
    piece the loop packs is known to fit."""
    out = []
    buf = ""
    for piece in pieces:
        if split_oversize and byte_len(piece) > MAX_CHUNK_BYTES:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(split_oversize(piece))
            continue
        candidate = buf + sep + piece if buf else piece
        if byte_len(candidate) <= MAX_CHUNK_BYTES:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = piece
    if buf:
        out.append(buf)
    return out


def chunk_text(text):
    text = text.strip()
    if not text:
        return []
    if byte_len(text) <= MAX_CHUNK_BYTES:
        return [text]
    # Strip each sentence piece so a kept trailing newline can't turn the
    # space join into artifacts like "mundo\n hola".
    pieces = [p.strip() for p in re.split(r"(?<=[.!?。！？\n])\s*", text) if p.strip()]
    return pack(pieces, " ", split_oversize=hard_split)


def hard_split(piece):
    return pack(piece.split(), " ", split_oversize=split_by_chars)


def split_by_chars(token):
    return pack(list(token), "")


def translate_chunk(chunk, source_lang):
    source_code = MYMEMORY_LANG_CODES.get(source_lang, source_lang)
    params = {"q": chunk, "langpair": f"{source_code}|en"}
    if CONTACT_EMAIL:
        params["de"] = CONTACT_EMAIL

    resp = _http.get(MYMEMORY_URL, params=params, timeout=15)
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
    the common-language fallback, with duplicates removed. English goes last as
    the give-up case for input that mirrored on every real candidate."""
    seen = set()
    result = []
    for code in [*ranked, *FALLBACK_LANGS]:
        if code != "en" and code not in seen:
            seen.add(code)
            result.append(code)
    result.append("en")
    return result


def is_cognate_candidate(text):
    """One short word. Long enough input translates for real, so only tokens
    like "no" or "Hotel" get the benefit of the mirror-as-cognate rule."""
    text = text.strip()
    return " " not in text and len(text) < MAX_COGNATE_CHARS


def translate_single_chunk(chunk, ranked, confidence=None):
    """Translate a single-chunk input, trying candidates until one returns more
    than the input echoed back. A confidently detected single word may keep its
    echo, that's the cognate case. Reaching "en" means everything else mirrored,
    so the input is treated as English without a MyMemory call, which the API
    would reject for the en pair anyway. The candidate list always ends with
    "en", so the loop always returns (source_lang, translation)."""
    confidence = confidence or {}
    candidates = candidate_langs(ranked)[:MAX_DETECT_ATTEMPTS]
    if "en" not in candidates:
        candidates.append("en")  # free to append, the "en" branch never calls out
    for source_lang in candidates:
        if source_lang == "en":
            return "en", chunk
        translated = translate_chunk(chunk, source_lang)
        if not is_mirror(chunk, translated):
            return source_lang, translated
        if is_cognate_candidate(chunk) and confidence.get(source_lang, 0.0) >= MIRROR_CONFIDENCE:
            # An empty echo still counts as a mirror, so fall back to the input.
            return source_lang, translated.strip() or chunk


def split_paragraphs(text):
    """Split on blank-line boundaries, capturing the exact separator strings.
    Returns paragraphs at even indexes and separators at odd ones, so joining
    the list back together reproduces the input byte for byte."""
    return re.split(r"(\n\s*\n)", text)


def translate_paragraphs(text, source_lang, whole_chunks=None):
    """Translate paragraph by paragraph so blank lines survive the chunker,
    which joins chunks with spaces and would otherwise flatten them. Each
    paragraph goes through the same chunker as before, so chunk boundaries
    and quota cost inside a paragraph don't change. The route has already
    chunked the whole text once to count chunks, so a single-paragraph text
    reuses that result through whole_chunks instead of chunking again."""
    out = []
    parts = split_paragraphs(text)
    reuse = whole_chunks is not None and len(parts) == 1
    for i, part in enumerate(parts):
        if i % 2:
            out.append(part)  # separator, passes through untouched
        else:
            chunks = whole_chunks if reuse else chunk_text(part)
            out.append(" ".join(translate_chunk(c, source_lang) for c in chunks))
    return "".join(out).strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    # CORS-open health check so external monitors can read the status cross-origin.
    resp = jsonify(status="ok")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def already_english_response(text):
    """The input is (or gets treated as) English, so it comes back unchanged."""
    return jsonify({
        "detected_code": "en",
        "detected_name": "English",
        "already_english": True,
        "translation": text,
    })


@app.route("/api/translate", methods=["POST"])
def api_translate():
    if rate_limited(client_address()):
        return jsonify({"error": "Slow down a little. Try again in a minute."}), 429

    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Type something to translate."}), 400
    if byte_len(text) > MAX_INPUT_BYTES:
        return jsonify({"error": "That's too much text at once. Keep it under 5,000 bytes."}), 413

    try:
        guesses = detect_langs(text)
    except LangDetectException:
        guesses = []
    ranked = [guess.lang for guess in guesses]
    confidence = {guess.lang: guess.prob for guess in guesses}

    if ranked[:1] == ["en"]:
        return already_english_response(text)

    chunks = chunk_text(text)
    if len(chunks) != 1 and not ranked:
        # Multi-chunk input can't lean on the single-chunk candidate retries.
        return jsonify({"error": "Couldn't figure out the language. Try a longer phrase."}), 422
    try:
        if len(chunks) == 1:
            source_lang, translated = translate_single_chunk(chunks[0], ranked, confidence)
            if source_lang == "en":
                # Every candidate mirrored, so the detector misread English.
                return already_english_response(text)
        else:
            source_lang = ranked[0]
            translated = translate_paragraphs(text, source_lang, chunks)
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
