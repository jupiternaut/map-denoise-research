import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import memory


class VisibilityTests(unittest.TestCase):
    def message(self, role='user', phase=None, text='真正的问题', kinds=None):
        return {'type':'response_item', 'timestamp':'2026-09-30T00:00:00Z',
                'payload':{'type':'message','id':'msg_test','role':role,'phase':phase,
                           'content':[{'type':'input_text','text':text}],
                           'internal_chat_message_metadata_passthrough':{'content_item_kinds':kinds or ['user.text']}}}

    def test_visible_user_and_assistant(self):
        self.assertEqual(memory.visible_message(self.message())[0]['text'], '真正的问题')
        for phase in ('commentary', 'final', 'final_answer'):
            self.assertIsNotNone(memory.visible_message(self.message('assistant', phase))[0])

    def test_hidden_roles_phases_and_tools_excluded(self):
        for role in ('system','developer','tool'):
            self.assertIsNone(memory.visible_message(self.message(role))[0])
        self.assertIsNone(memory.visible_message(self.message('assistant','analysis'))[0])
        self.assertIsNone(memory.visible_message({'type':'response_item','payload':{'type':'function_call'}})[0])

    def test_automatic_context_not_a_user_utterance(self):
        for kind in memory.AUTO_KINDS:
            self.assertEqual(memory.visible_message(self.message(kinds=[kind]))[1], 'automatic_context')
        for text in ('<environment_context>abc', '<external_codex_apps_open_page>abc',
                     '# AGENTS.md instructions\n<environment_context>abc'):
            self.assertEqual(memory.visible_message(self.message(text=text))[1], 'automatic_context')

    def test_attachment_instruction_is_preserved_as_visible_evidence(self):
        text = 'AGENTS.md says do X; I disagree.'
        self.assertEqual(memory.visible_message(self.message(text=text))[0]['text'], text)

    def test_redaction_does_not_export_credentials(self):
        raw = 'a ghp_' + 'X' * 32 + ' b sk-' + 'Y' * 32
        clean, count = memory.redact(raw)
        self.assertEqual(count, 2)
        self.assertNotIn('X' * 32, clean)

    def test_prefix_hash_excludes_later_appends(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source'
            path.write_bytes(b'first\n')
            digest = memory.sha_file(path, 6)
            with path.open('ab') as handle:
                handle.write(b'later\n')
            self.assertEqual(digest, memory.sha_file(path, 6))
            self.assertNotEqual(digest, memory.sha_file(path))

    def record(self, source):
        return {'id':'TEST001','title':'实际验收夹具','date':'2026-09-30','track':'test',
                'question':'验证不可改写登记', 'did':['使用来源哈希'], 'not_done':['不声称科研收益'],
                'result':['夹具'], 'interpretation':['夹具'], 'decision':['仅测试'],
                'evidence_level':'artifact_checked','sources':[str(source)],'tags':['test'],
                'predecessors':[],'corrections':[],'rerun_condition':'测试变化',
                'claims':['测试行为'],'evaluation_scope':'unit fixture'}

    def test_registry_requires_full_evidence_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); item=root/'record.json'; item.write_text('{}')
            with patch.object(memory,'ROOT',root), self.assertRaises(ValueError):
                memory.register(item)

    def test_registry_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'evidence.md'; source.write_text('finished fact')
            item=root/'record.json'; record=self.record(source); item.write_text(json.dumps(record))
            with patch.object(memory,'ROOT',root), patch.object(memory,'base_inventory',return_value=[]):
                memory.register(item)
                first=(root/'registry/TEST001.json').read_bytes()
                memory.register(item)
                self.assertEqual(first,(root/'registry/TEST001.json').read_bytes())
                record['result']=['rewritten fact']; item.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    memory.register(item)
                self.assertEqual(first,(root/'registry/TEST001.json').read_bytes())

    def test_registry_cannot_reuse_sealed_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'evidence.md'; source.write_text('fact')
            item=root/'record.json'; item.write_text(json.dumps(self.record(source)))
            with patch.object(memory,'ROOT',root), patch.object(memory,'base_inventory',return_value=[{'id':'TEST001'}]), self.assertRaises(ValueError):
                memory.register(item)

    def test_report_image_requires_registered_hash_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'REPORT.md'
            image = root / 'plot.png'
            image.write_bytes(b'fixture image bytes')
            text = '![figure](plot.png)'
            checks = [{'path':str(image), 'sha256':memory.sha_file(image)}]
            with patch.object(memory, 'PUBLISHED', root / 'published'):
                self.assertEqual(memory.portable_report_images(text, source, []), text)
                out = memory.portable_report_images(text, source, checks)
                self.assertIn('../media/', out)
                self.assertEqual(next((root/'published/media').glob('*')).read_bytes(), image.read_bytes())
                image.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    memory.portable_report_images(text, source, checks)


if __name__ == '__main__':
    unittest.main()
