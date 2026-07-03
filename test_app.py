"""Tests for the self-correcting detection logic. MyMemory is never called.
Every test stubs translate_chunk so the suite runs offline and deterministically."""

import unittest
from unittest.mock import patch

import app


class _G:
    """Stand-in for a langdetect guess (carries a .lang attribute)."""
    def __init__(self, lang):
        self.lang = lang


class _FakeResponse:
    """Stand-in for a requests response with a canned JSON payload."""
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class NormalizeAndMirror(unittest.TestCase):
    def test_normalize_collapses_space_and_case(self):
        self.assertEqual(app.normalize("  Muy   BIEN "), "muy bien")

    def test_mirror_exact(self):
        self.assertTrue(app.is_mirror("Muy bien", "Muy bien"))

    def test_mirror_ignores_case_and_spacing(self):
        self.assertTrue(app.is_mirror("Muy Bien", "muy   bien"))

    def test_empty_output_is_mirror(self):
        self.assertTrue(app.is_mirror("Muy bien", "   "))

    def test_real_translation_is_not_mirror(self):
        self.assertFalse(app.is_mirror("Muy bien", "Very good"))


class CandidateLangs(unittest.TestCase):
    def test_drops_english_and_dedupes_keeping_order(self):
        result = app.candidate_langs(["de", "en", "es", "de"])
        self.assertNotIn("en", result)
        self.assertEqual(result[:2], ["de", "es"])

    def test_falls_back_to_common_languages(self):
        result = app.candidate_langs(["de"])
        self.assertEqual(result[0], "de")
        for code in app.FALLBACK_LANGS:
            if code != "de":
                self.assertIn(code, result)

    def test_empty_ranking_uses_fallback_only(self):
        self.assertEqual(app.candidate_langs([]), app.FALLBACK_LANGS)


class Chunking(unittest.TestCase):
    """Pins the current behavior of chunk_text, hard_split, and split_by_chars."""

    def assert_chunks_fit(self, chunks):
        for chunk in chunks:
            self.assertLessEqual(app.byte_len(chunk), app.MAX_CHUNK_BYTES)
            # A clean encode/decode round trip proves no chunk broke a codepoint.
            self.assertEqual(chunk.encode("utf-8").decode("utf-8"), chunk)

    def test_empty_string_gives_no_chunks(self):
        self.assertEqual(app.chunk_text(""), [])
        self.assertEqual(app.chunk_text("   \n  "), [])

    def test_exactly_at_limit_is_one_chunk(self):
        text = "a" * app.MAX_CHUNK_BYTES
        self.assertEqual(app.chunk_text(text), [text])

    def test_one_byte_over_limit_splits(self):
        text = "a" * (app.MAX_CHUNK_BYTES + 1)
        chunks = app.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assert_chunks_fit(chunks)
        self.assertEqual("".join(chunks), text)

    def test_sentences_pack_without_losing_words(self):
        text = " ".join(["El zorro salta sobre el perro."] * 40)
        chunks = app.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assert_chunks_fit(chunks)
        self.assertEqual(app.normalize(" ".join(chunks)), app.normalize(text))

    def test_multibyte_never_splits_mid_codepoint(self):
        # One ascii char up front so the byte cap lands mid-codepoint in the
        # run of three-byte characters that follows.
        text = "a" + "あ" * 400
        chunks = app.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assert_chunks_fit(chunks)
        self.assertEqual("".join(chunks), text)

    def test_giant_unbroken_token_splits_by_chars(self):
        token = "x" * 2000
        chunks = app.chunk_text(token)
        self.assertEqual(len(chunks), 5)
        self.assert_chunks_fit(chunks)
        self.assertEqual("".join(chunks), token)

    def test_hard_split_packs_words(self):
        piece = " ".join(["palabra"] * 200)
        out = app.hard_split(piece)
        self.assertGreater(len(out), 1)
        self.assert_chunks_fit(out)
        self.assertEqual(" ".join(out).split(), piece.split())

    def test_split_by_chars_multibyte(self):
        token = "é" * 500  # two bytes each
        out = app.split_by_chars(token)
        self.assertGreater(len(out), 1)
        self.assert_chunks_fit(out)
        self.assertEqual("".join(out), token)


class TranslateChunkResponseShape(unittest.TestCase):
    """MyMemory sometimes returns a string in responseData instead of a dict."""

    def test_string_response_data_raises_service_error(self):
        payload = {"responseStatus": 200, "responseData": "PLEASE SELECT TWO DISTINCT LANGUAGES"}
        with patch.object(app.requests, "get", return_value=_FakeResponse(payload)):
            with self.assertRaises(RuntimeError):
                app.translate_chunk("hola", "es")

    def test_dict_response_data_returns_translation(self):
        payload = {"responseStatus": 200, "responseData": {"translatedText": "hello"}}
        with patch.object(app.requests, "get", return_value=_FakeResponse(payload)):
            self.assertEqual(app.translate_chunk("hola", "es"), "hello")


class TranslateShort(unittest.TestCase):
    def test_skips_mirror_and_returns_first_real_translation(self):
        table = {("Muy Bien", "de"): "Muy bien", ("Muy Bien", "es"): "Very good"}
        with patch.object(app, "candidate_langs", return_value=["de", "es"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: table[(c, l)]):
            lang, out = app.translate_short("Muy Bien", [])
        self.assertEqual(lang, "es")
        self.assertEqual(out, "Very good")

    def test_returns_none_when_every_candidate_mirrors(self):
        with patch.object(app, "candidate_langs", return_value=["de", "es", "fr"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c):
            lang, out = app.translate_short("Muy Bien", [])
        self.assertIsNone(lang)
        self.assertIsNone(out)

    def test_first_candidate_wins_when_it_translates(self):
        with patch.object(app, "candidate_langs", return_value=["es", "de"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: "Very good"):
            lang, out = app.translate_short("Muy Bien", [])
        self.assertEqual(lang, "es")

    def test_honors_attempt_cap(self):
        calls = []
        with patch.object(app, "candidate_langs", return_value=[f"l{i}" for i in range(20)]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: calls.append(l) or c):
            app.translate_short("xx", [])
        self.assertEqual(len(calls), app.MAX_DETECT_ATTEMPTS)


class ApiTranslate(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        app._rate_buckets.clear()  # each test starts with a fresh allowance

    def _post(self, text):
        return self.client.post("/api/translate", json={"text": text})

    def test_empty_input(self):
        self.assertEqual(self._post("   ").status_code, 400)

    def test_health_is_ok_and_cors_open(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")

    def test_english_short_circuits_without_calling_mymemory(self):
        with patch.object(app, "translate_chunk") as tc:
            r = self._post("Hello, how are you doing today my friend?")
            tc.assert_not_called()
        body = r.get_json()
        self.assertTrue(body["already_english"])
        self.assertEqual(body["detected_code"], "en")

    def test_muy_bien_self_corrects_to_spanish(self):
        table = {("Muy Bien", "de"): "Muy bien", ("Muy Bien", "es"): "Very good"}
        with patch.object(app, "candidate_langs", return_value=["de", "es"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: table[(c, l)]):
            r = self._post("Muy Bien")
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(body["detected_code"], "es")
        self.assertEqual(body["detected_name"], "Spanish")
        self.assertEqual(body["translation"], "Very good")

    def test_undetectable_short_phrase_returns_422(self):
        with patch.object(app, "detect_langs", return_value=[_G("de")]), \
             patch.object(app, "candidate_langs", return_value=["de", "es"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c):
            r = self._post("zzz")
        self.assertEqual(r.status_code, 422)

    def test_long_input_detects_once_and_translates_each_chunk(self):
        with patch.object(app, "detect_langs", return_value=[_G("fr")]), \
             patch.object(app, "chunk_text", return_value=["a", "b"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: "T") as tc:
            r = self._post("long french text stub")
        body = r.get_json()
        self.assertEqual(body["detected_code"], "fr")
        self.assertEqual(body["translation"], "T T")
        self.assertEqual(tc.call_count, 2)

    def test_upstream_error_returns_fixed_502_message(self):
        with patch.object(app, "detect_langs", return_value=[_G("de")]), \
             patch.object(app, "candidate_langs", return_value=["de"]), \
             patch.object(app, "translate_chunk", side_effect=RuntimeError("USED ALL FREE TRANSLATIONS")):
            r = self._post("Hallo")
        self.assertEqual(r.status_code, 502)
        # The upstream detail stays in the log, never in the response body.
        self.assertEqual(
            r.get_json()["error"],
            "The translation service returned an error. Try again in a minute.",
        )


class InputSizeCap(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        app._rate_buckets.clear()

    def _post(self, text):
        # English detection short-circuits before any MyMemory call.
        with patch.object(app, "detect_langs", return_value=[_G("en")]):
            return self.client.post("/api/translate", json={"text": text})

    def test_exactly_at_the_limit_passes(self):
        r = self._post("a" * app.MAX_INPUT_BYTES)
        self.assertEqual(r.status_code, 200)

    def test_over_the_limit_returns_413(self):
        r = self._post("a" * (app.MAX_INPUT_BYTES + 1))
        self.assertEqual(r.status_code, 413)
        self.assertIn("too much text", r.get_json()["error"])


class RateLimiting(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        app._rate_buckets.clear()

    def _post(self):
        # English detection short-circuits before any MyMemory call.
        with patch.object(app, "detect_langs", return_value=[_G("en")]):
            return self.client.post("/api/translate", json={"text": "hello there"})

    def test_allows_up_to_the_limit(self):
        for _ in range(app.RATE_LIMIT_MAX):
            self.assertEqual(self._post().status_code, 200)

    def test_over_the_limit_returns_429(self):
        for _ in range(app.RATE_LIMIT_MAX):
            self._post()
        r = self._post()
        self.assertEqual(r.status_code, 429)
        self.assertIn("error", r.get_json())

    def test_window_expiry_frees_the_bucket(self):
        clock = [0.0]
        with patch.object(app, "_now", side_effect=lambda: clock[0]):
            for _ in range(app.RATE_LIMIT_MAX):
                self._post()
            self.assertEqual(self._post().status_code, 429)
            clock[0] = float(app.RATE_LIMIT_WINDOW + 1)
            self.assertEqual(self._post().status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
