# Read into English

A tiny web app. Type text in any language on the left, and it auto-detects the
language and shows you the English on the right. Powered by the free
[MyMemory](https://mymemory.translated.net/) API.

## How it works

MyMemory's API **can't** auto-detect a source language. It rejects `auto` with
a 403. So this app does it in two steps.

1. Detect the language locally with `langdetect` (free, offline, no limit).
2. Translate from the detected language into English via MyMemory, splitting
   long input into <500-byte chunks because that's MyMemory's per-request cap.

Your input goes to *your* server first, which then calls MyMemory, so there's no
API key sitting in the browser.

### Self-correcting detection on short input

Detecting one or two words is unreliable for every offline detector, so for
short input the app uses the translation itself as a tie-breaker. When MyMemory
has no translation for the source language it was given, it echoes the input
straight back. The app treats that mirrored result as a signal the guess was
wrong, then tries the next candidate language until one returns something
different. If none do, it reports that it couldn't identify the language instead
of showing a wrong answer.

Worked example: `Muy Bien` is detected as German, `de|en` echoes back `Muy bien`
(a mirror), so the app falls through to Spanish, `es|en` returns `Very good`, and
the badge corrects to Spanish.

## Run it

```bash
# 1. (optional) create a virtual environment
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the server
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## Good to know

- The daily limit for anonymous use is about 5,000 words/day. Setting
  `MYMEMORY_EMAIL` raises it to roughly 50,000/day. When you hit the cap,
  the app shows MyMemory's "used all free translations for today" message.
- Detection on short text is imperfect. A two- or three-word phrase in
  Russian can be misread as another Cyrillic language. The self-correcting step
  above recovers many of these, but longer input still detects more reliably.
  This is a limitation of free detection, not a bug.
- Short input can cost extra quota. When the first guess mirrors, the app
  retries other languages, up to six MyMemory calls for one ambiguous phrase.
  Correct first guesses still cost one call. The cap lives in
  `MAX_DETECT_ATTEMPTS`.
- Quality comes from MyMemory's translation-memory matches plus machine
  translation fallback. It's strong on common phrases, weaker on rare ones.

## Deploy

The repo ships three deploy paths. The `Dockerfile` builds an image that runs
gunicorn on port 8080 as a non-root user. `deploy/translator.service` is a
systemd unit that runs gunicorn on port 5000 from a project virtualenv, so edit
the placeholders at the top before installing it. `render.yaml` deploys to
Render's free tier and reads `MYMEMORY_EMAIL` from the dashboard.
