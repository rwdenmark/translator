# Translate to English

A tiny web app. Type text in any language on the left, and it auto-detects the
language and shows you the English on the right. Powered by the free
[MyMemory](https://mymemory.translated.net/) API.

## How it works

MyMemory's API **can't** auto-detect a source language. It rejects `auto` with
a 403. So this app does it in two steps.

1. Detect the language locally with `langdetect` (free, offline, no limit).
2. Translate from the detected language into English via MyMemory, splitting
   long input into <500-byte chunks because that's MyMemory's per-request cap.
   Long input goes paragraph by paragraph, so blank lines survive translation.

Your input goes to *your* server first, which then calls MyMemory. MyMemory
doesn't use API keys, so there's no secret to protect. The server hop exists so
detection, chunking, and rate limiting all live in one place.

### Self-correcting detection on short input

Detecting one or two words is unreliable for every offline detector, so for
short input the app uses the translation itself as a tie-breaker. When MyMemory
has no translation for the source language it was given, it echoes the input
straight back. The app treats that mirrored result as a signal the guess was
wrong, then tries the next candidate language until one returns something
different. Two escape hatches keep real words from getting stuck in that loop.
A single short word that mirrors under a confident detection is accepted as a
cognate, so "no" and "Hotel" come back as themselves. And if every candidate
mirrors, the input is treated as English, which is what a mirrored short
English phrase usually is.

Worked example. `Muy Bien` is detected as German, `de|en` echoes back `Muy bien`
(a mirror), so the app falls through to Spanish, `es|en` returns `Very good`, and
the badge corrects to Spanish.

## Run it

```bash
# 1. (optional) create a virtual environment
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. install the pinned dependencies
pip install -r requirements.txt

# 3. start the server
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## Good to know

- The daily limit for anonymous use is about 5,000 chars/day. Setting
  `MYMEMORY_EMAIL` raises it to roughly 50,000 chars/day. When you hit the cap,
  the app shows its fixed try-again message and logs the MyMemory detail.
- One request takes at most 5,000 bytes of UTF-8, and each address gets 10
  requests a minute. Both caps keep a single visitor from burning the shared
  daily quota.
- Detection on short text is imperfect. A two- or three-word phrase in
  Russian can be misread as another Cyrillic language. The self-correcting step
  above recovers many of these, but longer input still detects more reliably.
- Short input can cost extra quota. When the first guess mirrors, the app
  retries other languages, up to six MyMemory calls for one ambiguous phrase.
  Correct first guesses still cost one call. The cap lives in
  `MAX_DETECT_ATTEMPTS`.
- Translation quality is strongest on common phrases and weaker on rare ones.

## Deploy

The repo ships three deploy paths. The `Dockerfile` builds an image that runs
gunicorn on port 8080 as a non-root user. `deploy/translator.service` is a
systemd unit that runs gunicorn on port 8085 from a project virtualenv, so edit
the placeholders at the top before installing it. `render.yaml` deploys to
Render's free tier, health-checks `/api/health`, and reads `MYMEMORY_EMAIL`
from the dashboard.
