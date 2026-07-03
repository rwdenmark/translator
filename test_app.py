"""Tests for the self-correcting detection logic. MyMemory is never called.
Every test stubs translate_chunk so the suite runs offline and deterministically."""

import unittest
from unittest.mock import patch

import app


class _G:
    """Stand-in for a langdetect guess (carries a .lang attribute)."""
    def __init__(self, lang):
        self.lang = lang


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

    def test_quota_error_surfaces_as_502(self):
        with patch.object(app, "detect_langs", return_value=[_G("de")]), \
             patch.object(app, "candidate_langs", return_value=["de"]), \
             patch.object(app, "translate_chunk", side_effect=RuntimeError("USED ALL FREE TRANSLATIONS")):
            r = self._post("Hallo")
        self.assertEqual(r.status_code, 502)
        self.assertIn("FREE", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
