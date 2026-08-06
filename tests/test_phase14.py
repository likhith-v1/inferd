from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicSlidingWindowLayer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

from bench.metrics import write_result_json
from bench.model_loader import load
from bench.pair_configs import PairConfig, validate_local_revisions
from bench.run_all import latest_phase14, latest_spec
from bench.correctness import _familywise_max_test
from core.model_runner import ModelRunner
from core.speculation import CROP_NO_REPLAY, RESTORE_AND_REPLAY


def _dense_cache(length: int) -> DynamicCache:
    cache = DynamicCache()
    values = torch.zeros(1, 1, length, 2)
    cache.update(values, values.clone(), 0)
    return cache


def _runner(mode: str) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cpu"
    runner.cache_reconciliation = mode
    return runner


class CacheReconciliationTests(unittest.TestCase):
    def test_dense_reconciliation_crops_and_forwards_only_final_token(self):
        for accepted, emitted in ((0, [90]), (2, [10, 11, 90]), (4, [10, 11, 12, 13, 90])):
            with self.subTest(accepted=accepted):
                runner = _runner(CROP_NO_REPLAY)
                cache = _dense_cache(5)
                checkpoint = runner.checkpoint_speculation(cache)
                extra = torch.ones(1, 1, 4, 2)
                cache.update(extra, extra.clone(), 0)
                widths = []

                def forward(tokens, kv):
                    widths.append(tokens.shape[1])
                    row = torch.ones(1, 1, tokens.shape[1], 2)
                    kv.update(row, row.clone(), 0)
                    return torch.zeros(1, tokens.shape[1], 3), kv

                runner.forward = forward
                _, cache = runner.reconcile_speculation(cache, checkpoint, accepted, emitted)
                self.assertEqual(widths, [1])
                self.assertEqual(cache.get_seq_length(), 5 + accepted + 1)

    def test_hybrid_restores_snapshot_and_replays_every_emitted_token(self):
        config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention"], num_hidden_layers=2
        )
        cache = Qwen3_5DynamicCache(config)
        cache.conv_states[0] = torch.ones(1, 2, 2)
        cache.recurrent_states[0] = torch.full((1, 2, 2), 2.0)
        cache.key_cache[1] = torch.zeros(1, 1, 5, 2)
        cache.value_cache[1] = torch.zeros(1, 1, 5, 2)
        runner = _runner(RESTORE_AND_REPLAY)
        checkpoint = runner.checkpoint_speculation(cache)
        cache.conv_states[0].fill_(9)
        cache.recurrent_states[0].fill_(9)
        cache.key_cache[1] = torch.zeros(1, 1, 9, 2)
        cache.value_cache[1] = torch.zeros(1, 1, 9, 2)
        observed = {}

        def forward(tokens, kv):
            observed["width"] = tokens.shape[1]
            observed["conv"] = kv.conv_states[0].clone()
            return torch.zeros(1, tokens.shape[1], 3), kv

        runner.forward = forward
        runner.reconcile_speculation(cache, checkpoint, 2, [10, 11, 90])
        self.assertEqual(observed["width"], 3)
        torch.testing.assert_close(observed["conv"], torch.ones(1, 2, 2))
        self.assertEqual(cache.key_cache[1].shape[-2], 5)

    def test_unknown_and_malformed_caches_fail_closed(self):
        runner = _runner(CROP_NO_REPLAY)
        with self.assertRaises(TypeError):
            runner.checkpoint_speculation(object())
        malformed = _dense_cache(2)
        malformed.layers[0].values = torch.zeros(1, 1, 3, 2)
        with self.assertRaises(ValueError):
            runner.checkpoint_speculation(malformed)

        sliding = DynamicCache()
        sliding.layers = [DynamicSlidingWindowLayer(4)]
        values = torch.zeros(1, 1, 2, 2)
        sliding.layers[0].update(values, values.clone())
        with self.assertRaises(TypeError):
            runner.checkpoint_speculation(sliding)

        with self.assertRaises(TypeError):
            _runner(RESTORE_AND_REPLAY).checkpoint_speculation(_dense_cache(2))


class CorrectnessGateTests(unittest.TestCase):
    def test_familywise_max_test_rejects_only_outside_joint_null(self):
        null_a = [float(i) for i in range(200)]
        null_b = list(reversed(null_a))
        self.assertTrue(_familywise_max_test([(198.5, null_a), (100.0, null_b)])[0])
        self.assertFalse(_familywise_max_test([(250.0, null_a), (100.0, null_b)])[0])


class LoaderRoutingTests(unittest.TestCase):
    def _load_with_type(self, model_type: str):
        tokenizer = Mock()
        if model_type == "qwen3":
            model = SimpleNamespace(model=object(), lm_head=object(), eval=lambda: None)
        else:
            model = SimpleNamespace(
                model=SimpleNamespace(language_model=object()),
                lm_head=object(),
                eval=lambda: None,
            )
        with tempfile.TemporaryDirectory() as directory, \
             patch("bench.model_loader.AutoConfig.from_pretrained",
                   return_value=SimpleNamespace(model_type=model_type)), \
             patch("bench.model_loader.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("bench.model_loader.AutoModelForCausalLM.from_pretrained", return_value=model) as causal, \
             patch("bench.model_loader.AutoModelForMultimodalLM.from_pretrained", return_value=model) as multimodal:
            loaded = load(directory, device="cpu")
        return loaded, causal, multimodal

    def test_qwen3_uses_causal_loader(self):
        _, causal, multimodal = self._load_with_type("qwen3")
        causal.assert_called_once()
        multimodal.assert_not_called()

    def test_qwen35_uses_multimodal_loader(self):
        _, causal, multimodal = self._load_with_type("qwen3_5")
        multimodal.assert_called_once()
        causal.assert_not_called()

    def test_unknown_model_type_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("bench.model_loader.AutoConfig.from_pretrained",
                   return_value=SimpleNamespace(model_type="unknown")):
            with self.assertRaises(ValueError):
                load(directory, device="cpu")


class PairValidationTests(unittest.TestCase):
    def _pair_runner(self, vocab, *, eos=2):
        runner = _runner(CROP_NO_REPLAY)
        runner.lm = SimpleNamespace(config=SimpleNamespace(vocab_size=len(vocab)))
        runner.lm_head = SimpleNamespace(out_features=len(vocab))
        runner.tokenizer = MagicMock()
        runner.tokenizer.get_vocab.return_value = vocab
        runner.tokenizer.get_added_vocab.return_value = {}
        runner.tokenizer.bos_token_id = 1
        runner.tokenizer.eos_token_id = eos
        runner.tokenizer.pad_token_id = 0
        runner.tokenizer.__len__.return_value = len(vocab)
        runner.tokenizer.name_or_path = ""
        runner._validated_drafts = set()
        runner._tokenizer_sha256 = None
        return runner

    def test_tokenizer_mismatch_fails_before_speculation(self):
        target = self._pair_runner({"a": 0, "b": 1})
        draft = self._pair_runner({"a": 0, "b": 2})
        with self.assertRaises(ValueError):
            target.validate_speculation_pair(draft)

    def test_added_special_and_logit_mismatches_fail(self):
        for mismatch in ("added", "special", "logits"):
            with self.subTest(mismatch=mismatch):
                target = self._pair_runner({"a": 0, "b": 1})
                draft = self._pair_runner({"a": 0, "b": 1})
                if mismatch == "added":
                    draft.tokenizer.get_added_vocab.return_value = {"<x>": 1}
                elif mismatch == "special":
                    draft.tokenizer.eos_token_id = 9
                else:
                    draft.lm_head.out_features = 3
                with self.assertRaises(ValueError):
                    target.validate_speculation_pair(draft)

    def test_reserved_logit_slots_do_not_count_as_tokenizer_mismatch(self):
        target = self._pair_runner({"a": 0, "b": 1})
        draft = self._pair_runner({"a": 0, "b": 1})
        target.lm_head.out_features = 3
        draft.lm_head.out_features = 3
        target.validate_speculation_pair(draft, expected_vocab_size=3)


class ProvenanceAndPersistenceTests(unittest.TestCase):
    def test_missing_pinned_weights_name_the_download_command(self):
        with tempfile.TemporaryDirectory() as directory:
            pair = PairConfig(
                "test", str(Path(directory) / "missing-target"),
                str(Path(directory) / "missing-draft"), "org/target", "org/draft",
                "target-rev", "draft-rev", None, None, CROP_NO_REPLAY,
            )
            with self.assertRaisesRegex(
                FileNotFoundError,
                r"hf download org/target --revision target-rev --local-dir",
            ):
                validate_local_revisions(pair)

    def test_revision_metadata_must_match_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("target", "draft"):
                metadata = root / name / ".cache" / "huggingface" / "download"
                metadata.mkdir(parents=True)
                (metadata / "config.json.metadata").write_text("expected\nblob\ntime\n")
            pair = PairConfig(
                "test", str(root / "target"), str(root / "draft"), "t", "d",
                "expected", "expected", None, None, CROP_NO_REPLAY,
            )
            self.assertEqual(
                validate_local_revisions(pair), {"target": "expected", "draft": "expected"}
            )
            pair = PairConfig(
                "test", str(root / "target"), str(root / "draft"), "t", "d",
                "wrong", "expected", None, None, CROP_NO_REPLAY,
            )
            with self.assertRaises(ValueError):
                validate_local_revisions(pair)

    def test_result_writer_never_overwrites_same_second(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("bench.metrics.time.strftime", return_value="same"):
            first = write_result_json({"run": 1}, "phase14", Path(directory))
            second = write_result_json({"run": 2}, "phase14", Path(directory))
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text()), {"run": 1})
            self.assertEqual(json.loads(second.read_text()), {"run": 2})


class ResultSelectionTests(unittest.TestCase):
    def test_phase14_cannot_replace_historical_phase04_selector(self):
        phase14 = {"engine": "spec", "phase": 14, "pair_config": "phase14-full",
                   "draft_label": "stock"}
        phase04 = {"engine": "spec", "draft_label": "stock"}
        with patch("bench.run_all._all_results", return_value=[("new", phase14), ("old", phase04)]):
            self.assertIs(latest_spec("stock"), phase04)
            self.assertIs(latest_phase14("phase14-full"), phase14)


if __name__ == "__main__":
    unittest.main()
