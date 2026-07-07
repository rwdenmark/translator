"""Tests for the self-correcting detection logic. MyMemory is never called.
Every test stubs translate_chunk so the suite runs offline and deterministically."""

import unittest
from unittest.mock import patch

import app


class _G:
    """Stand-in for a langdetect guess (carries .lang and .prob attributes)."""
    def __init__(self, lang, prob=0.99):
        self.lang = lang
        self.prob = prob


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
    def test_dedupes_keeping_order_with_english_last(self):
        result = app.candidate_langs(["de", "en", "es", "de"])
        self.assertNotIn("en", result[:-1])
        self.assertEqual(result[-1], "en")
        self.assertEqual(result[:2], ["de", "es"])

    def test_falls_back_to_common_languages(self):
        result = app.candidate_langs(["de"])
        self.assertEqual(result[0], "de")
        for code in app.FALLBACK_LANGS:
            if code != "de":
                self.assertIn(code, result)

    def test_empty_ranking_uses_fallback_then_english(self):
        self.assertEqual(app.candidate_langs([]), app.FALLBACK_LANGS + ["en"])


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

    def test_newline_terminated_pieces_leave_no_join_artifacts(self):
        # Sentence pieces used to keep their trailing newline, so the space
        # join produced strings like "mundo\n hola".
        text = "hola mundo\n" * 60  # far over one chunk
        chunks = app.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertNotIn("\n ", " ".join(chunks))
        for chunk in chunks:
            self.assertEqual(chunk, chunk.strip())


class TranslateChunkResponseShape(unittest.TestCase):
    """MyMemory sometimes returns a string in responseData instead of a dict."""

    def test_string_response_data_raises_service_error(self):
        payload = {"responseStatus": 200, "responseData": "PLEASE SELECT TWO DISTINCT LANGUAGES"}
        with patch.object(app._http, "get", return_value=_FakeResponse(payload)):
            with self.assertRaises(RuntimeError):
                app.translate_chunk("hola", "es")

    def test_dict_response_data_returns_translation(self):
        payload = {"responseStatus": 200, "responseData": {"translatedText": "hello"}}
        with patch.object(app._http, "get", return_value=_FakeResponse(payload)):
            self.assertEqual(app.translate_chunk("hola", "es"), "hello")


class LangpairCasing(unittest.TestCase):
    """langdetect says zh-cn but MyMemory's langpair wants RFC3066 zh-CN."""

    def _capture_langpair(self, source_lang):
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured.update(params)
            return _FakeResponse({"responseStatus": 200, "responseData": {"translatedText": "hello"}})

        with patch.object(app._http, "get", side_effect=fake_get):
            app.translate_chunk("你好世界", source_lang)
        return captured["langpair"]

    def test_zh_cn_is_sent_with_rfc3066_casing(self):
        self.assertEqual(self._capture_langpair("zh-cn"), "zh-CN|en")

    def test_zh_tw_is_sent_with_rfc3066_casing(self):
        self.assertEqual(self._capture_langpair("zh-tw"), "zh-TW|en")

    def test_plain_codes_pass_through_unchanged(self):
        self.assertEqual(self._capture_langpair("es"), "es|en")


class TranslateSingleChunk(unittest.TestCase):
    def test_skips_mirror_and_returns_first_real_translation(self):
        table = {("Muy Bien", "de"): "Muy bien", ("Muy Bien", "es"): "Very good"}
        with patch.object(app, "candidate_langs", return_value=["de", "es"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: table[(c, l)]):
            lang, out = app.translate_single_chunk("Muy Bien", [])
        self.assertEqual(lang, "es")
        self.assertEqual(out, "Very good")

    def test_falls_through_to_english_when_every_candidate_mirrors(self):
        with patch.object(app, "candidate_langs", return_value=["de", "es", "fr"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c):
            lang, out = app.translate_single_chunk("Muy Bien", [])
        self.assertEqual(lang, "en")
        self.assertEqual(out, "Muy Bien")

    def test_first_candidate_wins_when_it_translates(self):
        with patch.object(app, "candidate_langs", return_value=["es", "de"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: "Very good"):
            lang, out = app.translate_single_chunk("Muy Bien", [])
        self.assertEqual(lang, "es")

    def test_honors_attempt_cap(self):
        calls = []
        with patch.object(app, "candidate_langs", return_value=[f"l{i}" for i in range(20)]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: calls.append(l) or c):
            app.translate_single_chunk("xx", [])
        self.assertEqual(len(calls), app.MAX_DETECT_ATTEMPTS)

    def test_confident_single_word_mirror_is_accepted_as_cognate(self):
        calls = []
        with patch.object(app, "translate_chunk", side_effect=lambda c, l: calls.append(l) or c):
            lang, out = app.translate_single_chunk("no", ["es"], {"es": 0.95})
        self.assertEqual(lang, "es")
        self.assertEqual(out, "no")
        self.assertEqual(calls, ["es"])  # one call, no retries burned

    def test_low_confidence_single_word_mirror_keeps_retrying(self):
        table = {("no", "es"): "no", ("no", "fr"): "not"}
        with patch.object(app, "candidate_langs", return_value=["es", "fr"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: table[(c, l)]):
            lang, out = app.translate_single_chunk("no", ["es"], {"es": 0.4})
        self.assertEqual(lang, "fr")
        self.assertEqual(out, "not")

    def test_multi_word_mirror_is_never_a_cognate(self):
        table = {("Muy Bien", "de"): "Muy bien", ("Muy Bien", "es"): "Very good"}
        with patch.object(app, "candidate_langs", return_value=["de", "es"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: table[(c, l)]):
            lang, out = app.translate_single_chunk("Muy Bien", ["de"], {"de": 0.99})
        self.assertEqual(lang, "es")
        self.assertEqual(out, "Very good")

    def test_long_single_token_mirror_is_never_a_cognate(self):
        token = "Grundstücksverkehr"  # over the cognate length cap
        with patch.object(app, "candidate_langs", return_value=["de"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c):
            lang, out = app.translate_single_chunk(token, ["de"], {"de": 0.99})
        self.assertEqual(lang, "en")  # fell through instead of trusting the echo


class ClientTestCase(unittest.TestCase):
    """Shared setup for every test class that posts through the Flask client."""

    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        app._rate_buckets.clear()  # every test starts with a fresh allowance


class ApiTranslate(ClientTestCase):
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

    def test_short_phrase_that_mirrors_everywhere_ends_as_english(self):
        # Misdetected short English used to dead-end in a 422. Now the English
        # fallback catches it and returns the input unchanged.
        with patch.object(app, "detect_langs", return_value=[_G("de", 0.5)]), \
             patch.object(app, "candidate_langs", return_value=["de", "es"]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c):
            r = self._post("hello there friend")
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(body["already_english"])
        self.assertEqual(body["detected_code"], "en")
        self.assertEqual(body["translation"], "hello there friend")

    def test_spanish_no_returns_no_in_one_call(self):
        with patch.object(app, "detect_langs", return_value=[_G("es", 0.9)]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c) as tc:
            r = self._post("no")
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(body["detected_code"], "es")
        self.assertEqual(body["translation"], "no")
        self.assertFalse(body["already_english"])
        self.assertEqual(tc.call_count, 1)

    def test_german_hotel_mirror_is_accepted(self):
        with patch.object(app, "detect_langs", return_value=[_G("de", 0.85)]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c) as tc:
            r = self._post("Hotel")
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(body["detected_code"], "de")
        self.assertEqual(body["detected_name"], "German")
        self.assertEqual(body["translation"], "Hotel")
        self.assertEqual(tc.call_count, 1)

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
        # The upstream detail goes to the log. The client gets the fixed message.
        self.assertEqual(
            r.get_json()["error"],
            "The translation service returned an error. Try again in a minute.",
        )


class Paragraphs(ClientTestCase):
    """Blank lines used to be flattened by the chunker's space join. Now the
    text splits on paragraph boundaries first and the separators come back."""

    LONG_PARA = ("El zorro salta sobre el perro. " * 20).strip()  # two chunks

    def test_split_keeps_exact_separators_and_round_trips(self):
        text = "uno\n\ndos\n \ntres\n\n\ncuatro"
        parts = app.split_paragraphs(text)
        self.assertEqual("".join(parts), text)
        self.assertEqual(parts[1::2], ["\n\n", "\n \n", "\n\n\n"])
        self.assertEqual(parts[0::2], ["uno", "dos", "tres", "cuatro"])

    def test_single_paragraph_output_matches_the_old_join(self):
        with patch.object(app, "translate_chunk", side_effect=lambda c, l: c.upper()):
            got = app.translate_paragraphs(self.LONG_PARA, "es")
            old = " ".join(c.upper() for c in app.chunk_text(self.LONG_PARA)).strip()
        self.assertEqual(got, old)

    def test_three_paragraphs_keep_blank_lines_at_no_extra_cost(self):
        text = "\n\n".join([self.LONG_PARA] * 3)
        per_para_chunks = len(app.chunk_text(self.LONG_PARA))
        self.assertGreater(per_para_chunks, 1)  # sanity, forces the long path
        with patch.object(app, "detect_langs", return_value=[_G("es")]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: "T") as tc:
            r = self.client.post("/api/translate", json={"text": text})
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        para_out = " ".join(["T"] * per_para_chunks)
        self.assertEqual(body["translation"], "\n\n".join([para_out] * 3))
        # Same chunker per paragraph, so the quota cost is still one call per chunk.
        self.assertEqual(tc.call_count, per_para_chunks * 3)

    def test_single_paragraph_request_chunks_only_once(self):
        # The route chunks to count, translate_paragraphs reuses that result.
        with patch.object(app, "detect_langs", return_value=[_G("es")]), \
             patch.object(app, "chunk_text", wraps=app.chunk_text) as ct, \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: "T"):
            r = self.client.post("/api/translate", json={"text": self.LONG_PARA})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(ct.call_count, 1)

    def test_mixed_double_and_single_newlines_survive(self):
        text = self.LONG_PARA + "\n\n" + "línea uno\nlínea dos"
        with patch.object(app, "detect_langs", return_value=[_G("es")]), \
             patch.object(app, "translate_chunk", side_effect=lambda c, l: c.upper()):
            r = self.client.post("/api/translate", json={"text": text})
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        first = " ".join(c.upper() for c in app.chunk_text(self.LONG_PARA))
        # The short paragraph fits one chunk, so its inner newline rides along.
        self.assertEqual(body["translation"], first + "\n\n" + "LÍNEA UNO\nLÍNEA DOS")


class InputSizeCap(ClientTestCase):
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
        # The cap is enforced in bytes, so the message says bytes.
        self.assertIn("bytes", r.get_json()["error"])


class RateLimiting(ClientTestCase):
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

    def test_idle_buckets_get_pruned(self):
        clock = [0.0]
        with patch.object(app, "_now", side_effect=lambda: clock[0]):
            app.rate_limited("10.0.0.1")
            clock[0] = float(app.RATE_LIMIT_WINDOW + 1)
            # Enough calls to guarantee at least one sweep runs.
            for i in range(app.RATE_PRUNE_EVERY):
                app.rate_limited(f"10.0.1.{i}")
        self.assertNotIn("10.0.0.1", app._rate_buckets)


class ProxyAddressing(ClientTestCase):
    """The limiter keys on the direct peer unless TRUST_PROXY is enabled."""

    def _post(self, headers=None):
        # English detection short-circuits before any MyMemory call.
        with patch.object(app, "detect_langs", return_value=[_G("en")]):
            return self.client.post(
                "/api/translate", json={"text": "hello there"}, headers=headers or {}
            )

    def test_forwarded_header_is_ignored_by_default(self):
        # Every request forges a different client, all land in the peer's bucket.
        for i in range(app.RATE_LIMIT_MAX):
            self._post(headers={"X-Forwarded-For": f"203.0.113.{i}"})
        r = self._post(headers={"X-Forwarded-For": "203.0.113.250"})
        self.assertEqual(r.status_code, 429)

    def test_trust_proxy_keys_on_rightmost_forwarded_hop(self):
        # The trusted proxy appends the true client last. The leftmost hop is
        # whatever the client sent, so forging it must not rotate the bucket.
        same_client = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}
        forged_prefix = {"X-Forwarded-For": "203.0.113.99, 198.51.100.1"}
        other_client = {"X-Forwarded-For": "203.0.113.1, 198.51.100.2"}
        with patch.object(app, "TRUST_PROXY", True):
            for _ in range(app.RATE_LIMIT_MAX):
                self.assertEqual(self._post(headers=same_client).status_code, 200)
            self.assertEqual(self._post(headers=same_client).status_code, 429)
            # A forged leftmost value lands in the same exhausted bucket.
            self.assertEqual(self._post(headers=forged_prefix).status_code, 429)
            # A different rightmost hop is a genuinely different client.
            self.assertEqual(self._post(headers=other_client).status_code, 200)

    def test_trust_proxy_without_header_falls_back_to_peer(self):
        with patch.object(app, "TRUST_PROXY", True):
            self.assertEqual(self._post().status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
