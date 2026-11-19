"""Tests for how diarization models are fetched and unpickled.

No network, GPU or real model: pyannote's Pipeline.from_pretrained is
replaced with a fake, and the torch.load checks use tiny checkpoints
written here. Run with:  python -m unittest discover -s tests
"""

import io
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

from scrivox.core import constants

try:
    import torch

    from scrivox.core import diarizer, torch_compat
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    from pyannote.audio import Pipeline  # noqa: F401
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    HAVE_PYANNOTE = True
except ImportError:
    HAVE_PYANNOTE = False

_EXECUTED = []


def _payload():
    _EXECUTED.append(True)
    return "ran"


class _Evil:
    """Unpickling this calls _payload(), standing in for os.system & co."""

    def __reduce__(self):
        return (_payload, ())


def _checkpoint(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return buf


class CustomModelTests(unittest.TestCase):
    def test_default_and_legacy_models_are_not_custom(self):
        for model in ("", "  ", constants.DEFAULT_DIARIZATION_MODEL,
                      " " + constants.DEFAULT_DIARIZATION_MODEL + " ",
                      "pyannote/speaker-diarization-3.1"):
            self.assertFalse(constants.is_custom_diarization_model(model), model)

    def test_other_ids_and_folders_are_custom(self):
        for model in ("someone/speaker-diarization-community-1", r"C:\models\mine",
                      "pyannote/speaker-diarization-community-1-cloud"):
            self.assertTrue(constants.is_custom_diarization_model(model), model)

    def test_revision_is_a_full_commit_hash(self):
        rev = constants.DEFAULT_DIARIZATION_REVISION
        self.assertRegex(rev, r"^[0-9a-f]{40}$")


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TorchLoadModeTests(unittest.TestCase):
    def setUp(self):
        _EXECUTED.clear()

    @unittest.skipUnless(HAVE_PYANNOTE, "needs pyannote.audio")
    def test_trusted_load_refuses_code_execution(self):
        buf = _checkpoint({"x": _Evil()})
        with torch_compat.pyannote_torch_load(trusted=True):
            # pyannote passes weights_only=False itself; the patch must win.
            with self.assertRaises(Exception):
                torch.load(buf, weights_only=False)
        self.assertEqual(_EXECUTED, [])

    @unittest.skipUnless(HAVE_PYANNOTE, "needs pyannote.audio")
    def test_trusted_load_accepts_pyannote_task_classes(self):
        spec = Specifications(problem=Problem.MULTI_LABEL_CLASSIFICATION,
                              resolution=Resolution.FRAME, duration=10.0,
                              classes=["a", "b"])
        buf = _checkpoint({"pyannote.audio": {"specifications": spec},
                           "state_dict": {"w": torch.zeros(2)}})
        with torch_compat.pyannote_torch_load(trusted=True):
            loaded = torch.load(buf, weights_only=False)
        self.assertEqual(loaded["pyannote.audio"]["specifications"], spec)

    def test_untrusted_load_allows_code_execution(self):
        buf = _checkpoint({"x": _Evil()})
        with torch_compat.pyannote_torch_load(trusted=False):
            loaded = torch.load(buf)
        self.assertEqual(loaded["x"], "ran")
        self.assertEqual(_EXECUTED, [True])

    def test_torch_load_is_restored(self):
        original = torch.load
        with torch_compat.pyannote_torch_load(trusted=False):
            self.assertIsNot(torch.load, original)
        self.assertIs(torch.load, original)

    def test_modes_never_overlap(self):
        """A trusted load waits for an unsafe one instead of sharing it."""
        entered_unsafe = threading.Event()
        release_unsafe = threading.Event()
        seen = []

        def unsafe():
            with torch_compat.pyannote_torch_load(trusted=False):
                entered_unsafe.set()
                release_unsafe.wait(5)

        def trusted():
            with torch_compat.pyannote_torch_load(trusted=True):
                seen.append(torch_compat._patch_mode)

        t1 = threading.Thread(target=unsafe)
        t1.start()
        entered_unsafe.wait(5)
        with mock.patch.object(torch_compat, "_register_safe_globals"):
            t2 = threading.Thread(target=trusted)
            t2.start()
            t2.join(0.3)
            self.assertTrue(t2.is_alive())  # still waiting for the unsafe load
            release_unsafe.set()
            t1.join(5)
            t2.join(5)
        self.assertEqual(seen, ["trusted"])


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class ResolveSnapshotTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.hub = self._tmp.name
        self.model_dir = os.path.join(
            self.hub, "models--pyannote--speaker-diarization-community-1")

    def test_pinned_revision_is_used_not_refs_main(self):
        rev = constants.DEFAULT_DIARIZATION_REVISION
        os.makedirs(os.path.join(self.model_dir, "snapshots", rev))
        os.makedirs(os.path.join(self.model_dir, "refs"))
        with open(os.path.join(self.model_dir, "refs", "main"), "w") as f:
            f.write("0" * 40)
        self.assertEqual(
            diarizer._resolve_snapshot(constants.DEFAULT_DIARIZATION_MODEL, self.hub, rev),
            os.path.join(self.model_dir, "snapshots", rev))

    def test_missing_pinned_revision_fails(self):
        os.makedirs(os.path.join(self.model_dir, "snapshots", "0" * 40))
        with self.assertRaises(FileNotFoundError):
            diarizer._resolve_snapshot(constants.DEFAULT_DIARIZATION_MODEL, self.hub,
                                       constants.DEFAULT_DIARIZATION_REVISION)


@unittest.skipUnless(HAVE_TORCH and HAVE_PYANNOTE, "needs torch and pyannote.audio")
class DiarizeLoadTests(unittest.TestCase):
    """diarize_audio picks the revision, the torch.load mode and the warning."""

    def setUp(self):
        import numpy as np
        import soundfile as sf
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.wav = os.path.join(self._tmp.name, "a.wav")
        sf.write(self.wav, np.zeros(1600, dtype="float32"), 16000)

    def run_diarize(self, model):
        fake_pipeline = mock.MagicMock()
        fake_pipeline.return_value.exclusive_speaker_diarization.itertracks.return_value = []
        modes = []

        def fake_from_pretrained(*args, **kwargs):
            modes.append(torch_compat._patch_mode)
            return fake_pipeline

        progress = []
        with mock.patch.object(diarizer, "_get_bundled_models_dir", return_value=None), \
                mock.patch("pyannote.audio.Pipeline.from_pretrained",
                           side_effect=fake_from_pretrained) as fp, \
                self.assertLogs(diarizer.log, "WARNING") if constants.is_custom_diarization_model(model) \
                else _no_logs():
            diarizer.diarize_audio(self.wav, "hf_x", diarization_model=model,
                                   on_progress=progress.append)
        return fp, modes, progress

    def test_default_model_is_pinned_and_trusted(self):
        fp, modes, progress = self.run_diarize(constants.DEFAULT_DIARIZATION_MODEL)
        fp.assert_called_once_with(constants.DEFAULT_DIARIZATION_MODEL,
                                   revision=constants.DEFAULT_DIARIZATION_REVISION,
                                   token="hf_x")
        self.assertEqual(modes, ["trusted"])
        self.assertFalse(any("WARNING" in p for p in progress))

    def test_legacy_model_is_upgraded_to_pinned_default(self):
        fp, modes, _ = self.run_diarize("pyannote/speaker-diarization-3.1")
        self.assertEqual(fp.call_args.kwargs["revision"],
                         constants.DEFAULT_DIARIZATION_REVISION)
        self.assertEqual(modes, ["trusted"])

    def test_custom_model_warns_and_loads_unsafe(self):
        fp, modes, progress = self.run_diarize("someone/other-model")
        fp.assert_called_once_with("someone/other-model", token="hf_x")
        self.assertEqual(modes, ["unsafe"])
        self.assertTrue(any("code execution allowed" in p for p in progress))


class _no_logs:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)
        self.addCleanup(sys.path.remove, root)
        import build
        self.build = build
        self.root = root

    def test_constants_load_without_torch(self):
        c = self.build.load_constants(self.root)
        self.assertEqual(c.DEFAULT_DIARIZATION_REVISION, constants.DEFAULT_DIARIZATION_REVISION)

    def test_download_is_pinned(self):
        with tempfile.TemporaryDirectory() as models, \
                mock.patch.dict(os.environ, {"HF_TOKEN": "hf_x"}), \
                mock.patch("huggingface_hub.snapshot_download") as dl:
            ok = self.build.download_diarization_models(
                models, constants.DEFAULT_DIARIZATION_MODEL,
                constants.DEFAULT_DIARIZATION_REVISION)
        self.assertTrue(ok)
        self.assertEqual(dl.call_args.kwargs["revision"], constants.DEFAULT_DIARIZATION_REVISION)


if __name__ == "__main__":
    unittest.main()
