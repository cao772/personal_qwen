"""CPU-only input-integrity checks; these do NOT run model inference or training."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import layer_train as lt


class CharacterTokenizer:
    """A synthetic template/character codec to exercise mask invariants only."""
    def apply_chat_template(self, messages, **kwargs):
        text = ''.join('<' + m['role'] + '>\n' + m['content'] + '\n</turn>\n'
                       for m in messages[:-1])
        return text + '<assistant>\n<think>\n\n</think>\n\n' + messages[-1]['content'] + '<END>'

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def example():
    return dict(id='synthetic-example', kind='synthetic_reading', user_approved=False,
                target_status='researcher_written_not_user_approved', loss_on='last_assistant_only',
                target_message_index=3, messages=[
                    {'role': 'user', 'content': 'The action is only planned.'},
                    {'role': 'assistant', 'content': 'It has already finished.'},
                    {'role': 'user', 'content': 'Correct the status.'},
                    {'role': 'assistant', 'content': 'It is planned, not confirmed complete.'}])


class PreparationChecks(unittest.TestCase):
    def test_previous_wrong_reply_is_context_only(self):
        row = example()
        result = lt.encode_final_answer(CharacterTokenizer(), row, 4096)
        supervised = ''.join(chr(t) for t in result['labels'] if t != -100)
        self.assertEqual(supervised, row['messages'][-1]['content'] + '<END>')
        self.assertNotIn('already finished', supervised)
        self.assertEqual(result['prior_assistant_turns'], 1)

    def test_empty_thinking_is_not_supervised(self):
        result = lt.encode_final_answer(CharacterTokenizer(), example(), 4096)
        supervised = ''.join(chr(t) for t in result['labels'] if t != -100)
        self.assertNotIn('<think>', supervised)
        self.assertNotIn('</think>', supervised)

    def test_overlong_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlong'):
            lt.encode_final_answer(CharacterTokenizer(), example(), 3)

    def test_research_cannot_masquerade_as_approval(self):
        row = example()
        row['user_approved'] = True
        with self.assertRaisesRegex(ValueError, 'attributed'):
            lt.admit_row(row)

    def test_grounded_target_requires_source(self):
        row = example()
        row['kind'] = 'grounded_reading'
        with self.assertRaisesRegex(ValueError, 'source'):
            lt.admit_row(row)

    def test_reasoning_targets_cannot_sneak_into_recipe(self):
        row = example()
        row['messages'][-1]['reasoning_content'] = 'Unreviewed reasoning'
        with self.assertRaisesRegex(ValueError, 'Reasoning'):
            lt.encode_final_answer(CharacterTokenizer(), row, 4096)

    def test_cross_boundary_token_merge_is_rejected(self):
        class MergingTokenizer(CharacterTokenizer):
            def encode(self, text, **kwargs):
                values = super().encode(text, **kwargs)
                if not text.endswith('<END>'):
                    values[-1] = 999999
                return values
        with self.assertRaisesRegex(ValueError, 'merged'):
            lt.encode_final_answer(MergingTokenizer(), example(), 4096)

    def test_dataset_modification_and_group_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = {'splits': {}}
            for name in ('train', 'dev', 'regression'):
                data = (json.dumps({'id': name, 'group_id': name}) + '\n').encode()
                (root / (name + '.jsonl')).write_bytes(data)
                manifest['splits'][name] = {'rows': 1, 'group_ids': [name],
                                          'sha256': hashlib.sha256(data).hexdigest()}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            lt.load_manifest(root, ['regression.jsonl'])
            overlap = copy.deepcopy(manifest)
            overlap['splits']['dev']['group_ids'] = ['train']
            (root / 'manifest.json').write_text(json.dumps(overlap))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                lt.load_manifest(root, ['regression.jsonl'])
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'dev.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                lt.load_manifest(root, ['regression.jsonl'])

    def test_missing_or_changed_locked_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Only checks manifest enforcement, never loads these placeholder bytes.
            assets = {'config.json': b'{}', 'tokenizer_config.json': b'{}',
                      'tokenizer.json': b'{}', 'model.safetensors': b'PLACEHOLDER-NOT-WEIGHTS',
                      'chat_template.jinja': b'SYNTHETIC-TEMPLATE'}
            for name, data in assets.items():
                (root / name).write_bytes(data)
            revision = 'a' * 40
            lock = {'model_id': 'openbmb/MiniCPM5-2B', 'revision': revision,
                    'files': {k: hashlib.sha256(v).hexdigest() for k, v in assets.items()}}
            (root / 'snapshot-lock.json').write_text(json.dumps(lock))
            lt.verify_snapshot(root, lock['model_id'], revision)
            (root / 'chat_template.jinja').unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                lt.verify_snapshot(root, lock['model_id'], revision)
            (root / 'chat_template.jinja').write_bytes(b'CHANGED')
            with self.assertRaisesRegex(ValueError, 'differ'):
                lt.verify_snapshot(root, lock['model_id'], revision)


if __name__ == '__main__':
    unittest.main()
