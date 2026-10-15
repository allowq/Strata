"""Opt-in bounded empty-answer retry and API failure contracts (requires patch C)."""
from __future__ import annotations

import threading
import unittest

from serve.frontend import CALL_START, CALL_END, THINK_END, OutputParser
from serve.test_code_span_quotes import SCHEMA, call_text, calls, visible


class ReasoningCloseRetry(unittest.TestCase):
    def run_service(self, thought, *, budget=0, retry=False, answer="```\nOK", sampling=None):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        engine = MockEngine(tok, [thought, answer])
        svc = Service(engine, tok, None)
        svc.reasoning_close_retry = retry
        events = list(svc.run([], True, [], 2000, {"reasoning_budget_tokens": budget, **(sampling or {})}, threading.Event()))
        return engine, [ev for kind, ev in events if kind == "event"], dict(events[-1][1])

    def test_retry_settles_a_pending_fence_close_before_continuing(self):
        engine, evs, done = self.run_service("Plan:\n```\nquoted " + THINK_END + "\n", retry=True)
        self.assertEqual(engine.turns, 2)
        self.assertEqual(visible(evs, "content"), "```\nOK")
        self.assertNotIn(THINK_END, visible(evs, "reasoning"))
        self.assertEqual(engine.tok.decode(engine.last_prompt).count(THINK_END), 1)

    def test_retry_continues_a_closed_thinking_block_with_no_answer(self):
        engine, evs, done = self.run_service("Settled.\n" + THINK_END + "\n\n", retry=True)
        self.assertEqual(engine.turns, 2)
        self.assertEqual(visible(evs, "content"), "```\nOK")

    def test_retry_does_not_duplicate_an_existing_close(self):
        engine, evs, done = self.run_service("Settled.\n" + THINK_END + "\n\n", retry=True)
        self.assertEqual(engine.tok.decode(engine.last_prompt).count(THINK_END), 1)

    def test_retry_remains_off_by_default(self):
        engine, evs, done = self.run_service("Settled.\n" + THINK_END + "\n\n")
        self.assertEqual(engine.turns, 1)
        self.assertEqual(visible(evs, "content"), "")

    def test_retry_does_not_replace_a_nonempty_answer(self):
        engine, evs, done = self.run_service("Settled.\n" + THINK_END + "\n\nAnswer", retry=True)
        self.assertEqual(engine.turns, 1)
        self.assertEqual(visible(evs, "content"), "Answer")

    def test_retry_settles_an_unmatched_quote_before_checking_for_an_answer(self):
        engine, evs, done = self.run_service("An unfinished `example " + THINK_END + "OK", retry=True,
                                            answer="Unnecessary continuation")
        self.assertEqual(engine.turns, 1)
        self.assertEqual(visible(evs, "content"), "OK")

    def test_retry_reuses_the_close_held_by_an_unmatched_quote(self):
        engine, evs, done = self.run_service("An unfinished `example " + THINK_END, retry=True, answer="OK")
        self.assertEqual(engine.turns, 2)
        self.assertEqual(engine.tok.decode(engine.last_prompt).count(THINK_END), 1)
        self.assertEqual(visible(evs, "content"), "OK")

    def test_retry_accepts_a_stop_prefix_from_the_first_generation(self):
        engine, evs, done = self.run_service("Settled.\n" + THINK_END + "\n\na", retry=True,
                                            sampling={"stop": "abc"}, answer="Unnecessary continuation")
        self.assertEqual(engine.turns, 1)
        self.assertEqual(visible(evs, "content"), "a")

    def test_retry_does_not_continue_a_call_waiting_for_natural_stop(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        engine = MockEngine(tok, ["Working.\n" + call_text(), "Unnecessary continuation"])
        svc = Service(engine, tok, None)
        svc.reasoning_close_retry = True
        events = list(svc.run([], True, SCHEMA, 2000, {}, threading.Event()))
        evs = [ev for kind, ev in events if kind == "event"]
        self.assertEqual(engine.turns, 1)
        self.assertEqual(len(calls(evs)), 1)
        self.assertEqual(visible(evs, "content"), "")

    def test_retry_treats_whitespace_as_an_empty_answer(self):
        engine, evs, done = self.run_service("Settled.\n" + THINK_END + "\n\n ", retry=True)
        self.assertEqual(engine.turns, 2)
        self.assertEqual(visible(evs, "content").strip(), "```\nOK")

    def test_retry_accepts_flushed_stop_prefix_as_answer(self):
        engine, evs, done = self.run_service("Settled.\n" + THINK_END + "\n\n", retry=True,
                                            answer="a", sampling={"stop": "abc"})
        self.assertEqual(engine.turns, 2)
        self.assertEqual(visible(evs, "content"), "a")

    def test_retry_is_limited_to_one_continuation(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        engine = MockEngine(tok, ["Settled.\n" + THINK_END + "\n\n", ""])
        svc = Service(engine, tok, None)
        svc.reasoning_close_retry = True
        with self.assertRaisesRegex(ValueError, "without an answer after one"):
            list(svc.run([], True, [], 2000, {}, threading.Event()))
        self.assertEqual(engine.turns, 2)

    def test_retry_does_not_accept_a_started_but_unfinished_call(self):
        from serve.server import ByteTokenizer, EmptyAnswerError, MockEngine, Service
        tok = ByteTokenizer()
        partial = CALL_START + "\n<function=noop>\n<parameter=/>\n</function>\n" + CALL_END
        engine = MockEngine(tok, ["Settled.\n" + THINK_END + "\n\n", partial])
        svc = Service(engine, tok, None)
        svc.reasoning_close_retry = True
        events = []
        with self.assertRaisesRegex(EmptyAnswerError, "unfinished tool call"):
            for event in svc.run([], True, SCHEMA, 2000, {}, threading.Event()):
                events.append(event)
        self.assertEqual(engine.turns, 2)
        self.assertTrue(any(kind == "event" and ev.kind == "tool_start" for kind, ev in events))
        self.assertFalse(any(kind == "done" for kind, _ in events))

    def test_partial_call_on_first_pass_is_not_repeated(self):
        from serve.server import ByteTokenizer, EmptyAnswerError, MockEngine, Service
        tok = ByteTokenizer()
        partial = THINK_END + "\n\n" + CALL_START + "\n<function=noop>\n<parameter=x>\nvalue"
        engine = MockEngine(tok, [partial, "Unnecessary continuation"])
        svc = Service(engine, tok, None)
        svc.reasoning_close_retry = True
        with self.assertRaisesRegex(EmptyAnswerError, "unfinished tool call"):
            list(svc.run([], True, SCHEMA, 2000, {}, threading.Event()))
        self.assertEqual(engine.turns, 1)

    def test_complete_auto_call_remains_an_answer_after_retry(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        engine = MockEngine(tok, ["Settled.\n" + THINK_END + "\n\n", call_text()])
        svc = Service(engine, tok, None)
        svc.reasoning_close_retry = True
        events = list(svc.run([], True, SCHEMA, 2000, {}, threading.Event()))
        self.assertEqual(engine.turns, 2)
        self.assertEqual(len(calls([ev for kind, ev in events if kind == "event"])), 1)
        self.assertEqual(events[-1][0], "done")

    def test_partial_call_keeps_the_default_behavior_when_retry_is_off(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        raw = THINK_END + "\n\n" + CALL_START + "\n<function=noop>\n<parameter=x>\nvalue"
        engine = MockEngine(tok, raw)
        events = list(Service(engine, tok, None).run([], True, SCHEMA, 2000, {}, threading.Event()))
        self.assertEqual(events[-1][0], "done")
        self.assertFalse(calls([ev for kind, ev in events if kind == "event"]))

    def test_content_does_not_make_an_unfinished_call_successful(self):
        from serve.server import ByteTokenizer, EmptyAnswerError, MockEngine, Service
        tok = ByteTokenizer()
        raw = THINK_END + "\n\nSome text.\n" + CALL_START + "\n<function=noop>\n<parameter=x>\nvalue"
        svc = Service(MockEngine(tok, raw), tok, None)
        svc.reasoning_close_retry = True
        with self.assertRaisesRegex(EmptyAnswerError, 'unfinished tool call'):
            list(svc.run([], True, SCHEMA, 2000, {}, threading.Event()))

    def test_incomplete_call_at_token_limit_retains_length_finish(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        raw = THINK_END + "\n\n" + CALL_START + "\n<function=noop>\n<parameter=x>\nvalue"
        svc = Service(MockEngine(tok, raw), tok, None)
        svc.reasoning_close_retry = True
        events = list(svc.run([], True, SCHEMA, len(tok.encode(raw)), {}, threading.Event()))
        self.assertEqual(events[-1][1]['finish'], 'length')

class EmptyAnswerApiError(unittest.TestCase):
    def check_api(self, api, stream, partial=False):
        import json
        import urllib.error
        import urllib.request
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, ROOT, Service, serve
        tok = ByteTokenizer()
        continuation = CALL_START + "\n<function=noop>\n<parameter=/>\n</function>\n" + CALL_END if partial else ""
        engine = MockEngine(tok, ["Settled." + THINK_END + "\n\n", continuation])
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.reasoning_close_retry = True
        httpd = serve(svc, port=0)
        body = dict(model="m", messages=[dict(role="user", content="2+2?")], max_tokens=2000, stream=stream)
        if partial:
            body['tools'] = [{'type': 'function', 'function': {'name': 'noop', 'parameters':
                             {'type': 'object', 'properties': {}}}}]
        if api == "messages":
            body["thinking"] = dict(type="enabled", budget_tokens=512)
            if partial:
                body['tools'] = [{'name': 'noop', 'input_schema': {'type': 'object', 'properties': {}}}]
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/{api}"
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            try:
                response = urllib.request.urlopen(request, timeout=10)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                raw = response.read().decode()
                self.assertEqual(response.status, 200 if stream else 500)
            self.assertIn("unfinished tool call" if partial else "without an answer after one", raw)
            self.assertIn("error", raw)
            self.assertNotIn('"finish_reason": "stop"', raw)
            self.assertNotIn('"stop_reason": "end_turn"', raw)
            self.assertEqual(engine.turns, 2)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_openai_stream(self):
        self.check_api("chat/completions", True)

    def test_openai_nonstream(self):
        self.check_api("chat/completions", False)

    def test_anthropic_stream(self):
        self.check_api("messages", True)

    def test_anthropic_nonstream(self):
        self.check_api("messages", False)

    def test_partial_openai_stream(self):
        self.check_api("chat/completions", True, partial=True)

    def test_partial_openai_nonstream(self):
        self.check_api("chat/completions", False, partial=True)

    def test_partial_anthropic_stream(self):
        self.check_api("messages", True, partial=True)

    def test_partial_anthropic_nonstream(self):
        self.check_api("messages", False, partial=True)


if __name__ == "__main__":
    unittest.main()
