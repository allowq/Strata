"""An early EOS must resume reasoning without synthesizing its closing marker."""
import threading
import unittest
import json
import urllib.request

from serve.frontend import THINK_END, OutputParser
from serve.server import ByteTokenizer, EmptyAnswerError, MockEngine, Service, StrataEngine
from serve.test_code_span_quotes import SCHEMA, call_text, calls, visible


class MaskAwareEngine(MockEngine):
    can_mask_tokens = True

    def __init__(self, tok, thought='A literal ``', continuation=None, answer='\n\nOK'):
        super().__init__(tok, '')
        self.thought = thought
        self.continuation = continuation if continuation is not None else THINK_END + '`` stays text.\n' + THINK_END
        self.answer = answer
        self.requests = []

    def generate(self, ids, max_new, sampling, cancel):
        self.requests.append((list(ids), dict(sampling or {})))
        turn = len(self.requests)
        text = self.thought if turn == 1 else (
            self.continuation if sampling.get('_strata_block_tokens') else call_text()
        ) if turn == 2 else self.answer
        self.script = self.tok.encode(text, parse_special=True) + self.tok.encode('<|im_end|>', parse_special=True)
        yield from super().generate(ids, max_new, sampling, cancel)


class ReasoningEosMask(unittest.TestCase):
    def test_debug_keeps_exact_eos_and_native_boundary_evidence(self):
        import ast
        import contextlib
        import io
        import os
        from unittest.mock import patch

        class RecordingEngine(MockEngine):
            can_mask_tokens = True

            def generate(self, *args, **kwargs):
                try:
                    for token in super().generate(*args, **kwargs):
                        self.observed.append(token)
                        yield token
                finally:
                    self.last = {'finish': 'stop'}

        tok = TokenAwareThinking.Tokenizer()
        engine = RecordingEngine(tok, ['A literal ``', THINK_END + '`` stays text.\n' + THINK_END, 'OK'])
        engine.observed = []
        service = Service(engine, tok, None)
        service.reasoning_close_retry = True
        output = io.StringIO()
        with patch.dict(os.environ, {'STRATA_DEBUG': '1'}), contextlib.redirect_stdout(output):
            result = list(service.run([], True, [], 500, {}, threading.Event()))
        records = dict(line.removeprefix('[strata] ').split(': ', 1)
                       for line in output.getvalue().splitlines() if line.startswith('[strata] raw'))
        self.assertEqual(ast.literal_eval(records['raw']), tok.decode(engine.observed))
        self.assertEqual(ast.literal_eval(records['raw token tail'].split(' stop_ids=')[0]), engine.observed[-32:])
        self.assertEqual(ast.literal_eval(records['raw reasoning-close token positions']),
                         [i for i, token in enumerate(engine.observed) if token == service.think_end_id])
        self.assertEqual(sum(token in service.stop_ids for token in engine.observed), 2)
        self.assertEqual(visible([ev for kind, ev in result if kind == 'event'], 'content'), 'OK')

    def test_loop_recovery_does_not_put_the_suppressed_eos_back_in_the_prompt(self):
        from serve.server import EFFORT_TEXT

        class RecordingEngine(MockEngine):
            can_mask_tokens = True

            def generate(self, ids, *args, **kwargs):
                self.prompts.append(list(ids))
                yield from super().generate(ids, *args, **kwargs)

        tok = ByteTokenizer()
        initial = 'A literal ``'
        engine = RecordingEngine(tok, [initial, 'Repeated analysis words for checking the result. ' * 700,
                                       '`` Done.\n' + THINK_END, 'OK'], max_context=100000)
        engine.prompts = []
        service = Service(engine, tok, None)
        service.reasoning_close_retry = True
        service.reasoning_loop_recovery = True
        head = '<|im_start|>system\n'
        ids = tok.encode(head + EFFORT_TEXT['xhigh'] + '<|im_end|>', parse_special=True)
        result = list(service.run(ids, True, [], 40000, {}, threading.Event()))
        events = [ev for kind, ev in result if kind == 'event']
        self.assertEqual(visible(events, 'content'), 'OK')
        self.assertEqual(result[-1][1]['reasoning_recoveries'], 1)
        low = tok.encode(head + EFFORT_TEXT['low'] + '<|im_end|>', parse_special=True)
        resumed = engine.prompts[2]
        self.assertEqual(resumed[:len(low)], low)
        generated = resumed[len(low):]
        self.assertFalse(set(generated) & service.stop_ids,
                         'the rejected EOS is diagnostic data, not part of the continued assistant turn')
        self.assertTrue(tok.decode(generated).startswith(initial + 'Repeated analysis'))

    def run_service(self, engine, *, max_new=2000, retry=True, **sampling):
        service = Service(engine, engine.tok, None)
        service.reasoning_close_retry = retry
        service.tool_call_recovery = True
        result = list(service.run([100], True, SCHEMA, max_new, sampling, threading.Event()))
        return service, [ev for kind, ev in result if kind == 'event'], result[-1][1]

    def test_preserves_literal_and_continues_reasoning_before_answering(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok)
        service, events, done = self.run_service(engine, temperature=0)
        self.assertEqual(visible(events, 'reasoning'), 'A literal ``</think>`` stays text.\n')
        self.assertEqual(visible(events, 'content').strip(), 'OK')
        self.assertFalse(calls(events))
        self.assertEqual(done['finish'], 'stop')
        self.assertEqual(len(engine.requests), 3)
        self.assertNotIn('_strata_block_tokens', engine.requests[0][1])
        self.assertEqual(set(engine.requests[1][1]['_strata_block_tokens']), service.stop_ids)
        self.assertNotIn('_strata_block_tokens', engine.requests[2][1])
        self.assertEqual(engine.requests[1][0], [100] + tok.encode(engine.thought, parse_special=True))

    def test_engine_wire_masks_are_explicit_and_default_wire_is_unchanged(self):
        self.assertEqual(StrataEngine.sampling_keys({}), '')
        self.assertEqual(StrataEngine.sampling_keys({'_strata_block_tokens': [248044, 248046]}),
                         ' ban0=248044 ban1=248046')

    def test_partial_marker_is_not_flushed_before_reasoning_resumes(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok, thought='A literal ``</thi',
                                 continuation='nk>`` stays text.\n' + THINK_END)
        _, events, _ = self.run_service(engine)
        self.assertEqual(visible(events, 'reasoning'), 'A literal ``</think>`` stays text.\n')
        self.assertEqual(visible(events, 'content').strip(), 'OK')
        self.assertFalse(calls(events))

    def test_complete_deferred_call_releases_the_mask_and_remains_a_real_call(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok, thought='Working.\n', continuation=call_text(), answer='')
        _, events, done = self.run_service(engine)
        self.assertEqual(len(calls(events)), 1)
        self.assertEqual(len(engine.requests), 3)
        self.assertNotIn('_strata_block_tokens', engine.requests[2][1])
        self.assertEqual(done['finish'], 'stop')

    def test_length_limit_during_recovery_does_not_start_another_pass(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok, thought='A', continuation='B' * 100)
        _, events, done = self.run_service(engine, max_new=12)
        self.assertEqual(len(engine.requests), 2)
        self.assertEqual(done['finish'], 'length')
        self.assertEqual(done['completion_tokens'], 12)
        self.assertEqual(visible(events, 'reasoning'), 'A' + 'B' * 10)

    def test_boundary_on_the_last_token_keeps_length_finish(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok, thought='A', continuation=THINK_END)
        limit = len(tok.encode('A', parse_special=True)) + 1 + len(tok.encode(THINK_END, parse_special=True))
        _, _, done = self.run_service(engine, max_new=limit)
        self.assertEqual(len(engine.requests), 2)
        self.assertEqual(done['finish'], 'length')

    def test_budget_wrap_releases_the_mask(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok, thought='A', continuation='B' * 100)
        _, events, done = self.run_service(engine, reasoning_budget_tokens=5)
        self.assertEqual(len(engine.requests), 3)
        self.assertNotIn('_strata_block_tokens', engine.requests[2][1])
        self.assertIn('time to give my answer.', visible(events, 'reasoning'))
        self.assertEqual(visible(events, 'content').strip(), 'OK')
        self.assertEqual(done['finish'], 'stop')

    def test_default_off_does_not_mask_or_retry(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok)
        _, events, done = self.run_service(engine, retry=False)
        self.assertEqual(len(engine.requests), 1)
        self.assertEqual(visible(events, 'reasoning'), engine.thought)
        self.assertEqual(visible(events, 'content'), '')
        self.assertEqual(done['finish'], 'stop')

    def test_older_engine_is_rejected_without_injecting_a_close(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok)
        engine.can_mask_tokens = False
        with self.assertRaisesRegex(EmptyAnswerError, 'token_mask support'):
            self.run_service(engine)
        self.assertEqual(len(engine.requests), 1)

    def test_broken_mask_capability_cannot_loop_or_silently_succeed(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok, thought='Still thinking.', continuation='')
        with self.assertRaisesRegex(EmptyAnswerError, 'suppressed stop token'):
            self.run_service(engine)
        self.assertEqual(len(engine.requests), 2)

    def test_wire_rejects_invalid_masks(self):
        for value in ('1 2', [1, 2, 3], [-1], [1.5], [True]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StrataEngine.sampling_keys({'_strata_block_tokens': value})

    def test_recovery_preserves_request_sampling(self):
        tok = ByteTokenizer()
        engine = MaskAwareEngine(tok)
        settings = {'temperature': 0.7, 'top_p': 0.83, 'top_k': 12, 'seed': 42}
        self.run_service(engine, **settings)
        for _, sampling in engine.requests:
            self.assertEqual({k: sampling[k] for k in settings}, settings)

    def test_both_apis_deliver_a_real_final_answer_after_masked_reasoning(self):
        from serve.frontend import ChatTemplate
        from serve.server import ROOT, serve
        for api in ('chat/completions', 'messages'):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    tok = TokenAwareThinking.Tokenizer()
                    engine = MaskAwareEngine(tok)
                    service = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
                    service.reasoning_close_retry = True
                    httpd = serve(service, port=0)
                    body = {'model': 'm', 'messages': [{'role': 'user', 'content': 'Answer OK.'}],
                            'max_tokens': 2000, 'stream': stream}
                    if api == 'messages':
                        body['thinking'] = {'type': 'enabled', 'budget_tokens': 512}
                    request = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/{api}',
                              data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    try:
                        with urllib.request.urlopen(request, timeout=10) as response:
                            raw = response.read().decode()
                            self.assertEqual(response.status, 200)
                        self.assertEqual(len(engine.requests), 3)
                        self.assertNotIn('"error"', raw)
                        self.assertNotIn('"type": "tool_use"', raw)
                        self.assertNotIn('"tool_calls": [', raw)
                        if not stream:
                            result = json.loads(raw)
                            if api == 'messages':
                                content = ''.join(x['text'] for x in result['content'] if x['type'] == 'text')
                                reasoning = ''.join(x['thinking'] for x in result['content'] if x['type'] == 'thinking')
                            else:
                                content = result['choices'][0]['message']['content']
                                reasoning = result['choices'][0]['message']['reasoning_content']
                            self.assertEqual(content.strip(), 'OK')
                            self.assertEqual(reasoning, 'A literal ``</think>`` stays text.\n')
                        else:
                            self.assertIn('message_stop' if api == 'messages' else '[DONE]', raw)
                            pieces = [json.loads(x[5:].strip()) for x in raw.splitlines()
                                      if x.startswith('data:') and x[5:].strip() != '[DONE]']
                            if api == 'messages':
                                content = ''.join(x.get('delta', {}).get('text', '') for x in pieces)
                                reasoning = ''.join(x.get('delta', {}).get('thinking', '') for x in pieces)
                            else:
                                deltas = [c.get('delta', {}) for x in pieces for c in x.get('choices', [])]
                                content = ''.join(x.get('content') or '' for x in deltas)
                                reasoning = ''.join(x.get('reasoning_content') or '' for x in deltas)
                            self.assertEqual(content.strip(), 'OK')
                            self.assertEqual(reasoning, 'A literal ``</think>`` stays text.\n')
                    finally:
                        httpd.shutdown()
                        httpd.server_close()



class TokenAwareThinking(unittest.TestCase):
    """The native thinking token and ordinary text with the same spelling differ."""
    class Tokenizer(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + [THINK_END]
        @property
        def special_tokens(self):
            return {THINK_END: 256 + self.SPECIALS.index(THINK_END)}

    def run_parts(self, parts, retry=True):
        tok = self.Tokenizer()
        engine = MockEngine(tok, "")
        # False means literal byte tokens, True means the vocabulary's thinking token.
        engine.script = [t for text, special in parts for t in tok.encode(text, parse_special=special)]
        engine.script += tok.encode("<|im_end|>", parse_special=True)
        service = Service(engine, tok, None)
        service.reasoning_close_retry = retry
        service.tool_call_recovery = True
        result = list(service.run([100], True, SCHEMA, 2000, {}, threading.Event()))
        return [ev for kind, ev in result if kind == "event"], result[-1][1]

    def test_plain_spelling_cannot_end_reasoning(self):
        thought = 'The literal spelling is </think>. Continue reasoning.\n'
        events, done = self.run_parts([(thought, False), (THINK_END, True), ('\nOK', False)])
        self.assertEqual(visible(events, 'reasoning'), thought)
        self.assertEqual(visible(events, 'content'), 'OK')
        self.assertFalse(calls(events))
        self.assertEqual(done['finish'], 'stop')

    def test_plain_marker_after_equal_width_runs_stays_text(self):
        thought = 'It would be `` ``</think>`` ``. Still reasoning.\n'
        events, _ = self.run_parts([(thought, False), (THINK_END, True), ('\nOK', False)])
        self.assertEqual(visible(events, 'reasoning'), thought)
        self.assertEqual(visible(events, 'content'), 'OK')

    def test_native_marker_inside_closed_quote_stays_text(self):
        before, after = 'A quoted ``', '`` remains reasoning.\n'
        events, _ = self.run_parts([(before, False), (THINK_END, True), (after, False),
                                    (THINK_END, True), ('\nOK', False)])
        self.assertEqual(visible(events, 'reasoning'), before + THINK_END + after)
        self.assertEqual(visible(events, 'content'), 'OK')

    def test_native_marker_after_escaped_first_tick_stays_in_reasoning(self):
        before, after = 'Example: \\``', '` stays reasoning.\n'
        events, done = self.run_parts([(before, False), (THINK_END, True), (after, False),
                                      (THINK_END, True), ('\nOK', False)])
        self.assertEqual(visible(events, 'reasoning'), before + THINK_END + after)
        self.assertEqual(visible(events, 'content'), 'OK')
        self.assertFalse(calls(events))
        self.assertEqual(done['finish'], 'stop')

    def test_native_close_then_genuine_tool_still_calls(self):
        events, done = self.run_parts([('Use the tool.\n', False), (THINK_END, True), (call_text(), True)])
        self.assertEqual(len(calls(events)), 1)
        self.assertEqual(calls(events)[0][0], 'noop')

    def test_limit_does_not_release_tool_actions_from_native_quote_lookahead(self):
        tok = self.Tokenizer()
        engine = MockEngine(tok, '')
        text = 'Plan:\n```xml\n' + THINK_END + '\n\nOK\n' + call_text()
        engine.script = tok.encode(text, parse_special=True)
        service = Service(engine, tok, None)
        service.reasoning_close_retry = True
        service.tool_call_recovery = True
        result = list(service.run([100], True, SCHEMA, len(engine.script), {}, threading.Event()))
        events = [ev for kind, ev in result if kind == 'event']
        self.assertEqual(result[-1][1]['finish'], 'length')
        self.assertFalse(any(e.kind.startswith('tool_') for e in events))
        self.assertEqual(visible(events, 'content'), 'OK\n' + call_text())

    def test_default_off_keeps_existing_text_parser(self):
        events, _ = self.run_parts([('Before </think>After', False)], retry=False)
        self.assertEqual(visible(events, 'reasoning'), 'Before ')
        self.assertEqual(visible(events, 'content'), 'After')

    def test_plain_marker_across_paragraphs_remains_literal(self):
        thought = 'A quote `\n</think>\n\n` still reasoning.\n\n'
        events, _ = self.run_parts([(thought, False), (THINK_END, True), ('\nOK', False)])
        self.assertEqual(visible(events, 'reasoning'), thought)
        self.assertEqual(visible(events, 'content'), 'OK')

    def test_pending_call_followed_by_plain_marker_is_reasoning(self):
        thought = call_text() + '</think> is quoted syntax.\n'
        events, _ = self.run_parts([(thought, False), (THINK_END, True), ('\nOK', False)])
        self.assertFalse(calls(events))
        self.assertEqual(visible(events, 'reasoning'), thought)
        self.assertEqual(visible(events, 'content'), 'OK')

    def test_plain_marker_inside_deferred_call_argument_is_data(self):
        events, _ = self.run_parts([(call_text('literal </think> value'), False), (THINK_END, True)])
        self.assertEqual(calls(events), [('noop', {'x': 'literal </think> value'})])

    def test_native_quote_metadata_survives_buffering_and_chunking(self):
        parts = [('Quoted ``', False), (THINK_END, True),
                 (' and literal </think> ', False), (THINK_END, True),
                 ('``. More reasoning.\n', False), (THINK_END, True), ('\nOK', False)]
        expected = ''.join(text for text, _ in parts[:-2])
        for width in (1, 3, 1000):
            parser = OutputParser(token_thinking=True)
            events = []
            for text, special in parts:
                if special:
                    events += parser.feed(text, think_end=True)
                else:
                    for i in range(0, len(text), width):
                        events += parser.feed(text[i:i+width])
            events += parser.finish()
            with self.subTest(width=width):
                self.assertEqual(visible(events, 'reasoning'), expected)
                self.assertEqual(visible(events, 'content'), 'OK')

    def test_explicit_budget_boundary_flushes_pending_native_quote(self):
        parser = OutputParser(token_thinking=True)
        events = parser.feed('Unfinished ``')
        events += parser.feed(THINK_END, think_end=True)
        events += parser.feed('Budget reached.\n' + THINK_END, protocol=True)
        events += parser.feed('\nOK')
        events += parser.finish()
        self.assertEqual(visible(events, 'reasoning'), 'Unfinished ``</think>Budget reached.\n')
        self.assertEqual(visible(events, 'content'), 'OK')

    def test_native_token_provenance_survives_masked_continuation(self):
        engine = MaskAwareEngine(self.Tokenizer())
        _, events, done = ReasoningEosMask().run_service(engine)
        self.assertEqual(visible(events, 'reasoning'), 'A literal ``</think>`` stays text.\n')
        self.assertEqual(visible(events, 'content').strip(), 'OK')
        self.assertFalse(calls(events))
        self.assertEqual(len(engine.requests), 3)

    def test_native_close_after_unfinished_code_still_ends_reasoning(self):
        events, _ = self.run_parts([('Unfinished `code', False), (THINK_END, True), ('\n\nOK', False)])
        self.assertEqual(visible(events, 'reasoning'), 'Unfinished `code')
        self.assertEqual(visible(events, 'content'), 'OK')

if __name__ == '__main__':
    unittest.main()
