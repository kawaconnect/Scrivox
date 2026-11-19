"""Unit tests for locating ffmpeg/ffprobe and for the ffprobe helpers.

The ffprobe tests run only when ffprobe and ffmpeg are installed.
Run with:  python -m unittest discover -s tests
"""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scrivox.core import media


def _make_tool(folder, name):
    exe = name + ".exe" if sys.platform == "win32" else name
    path = os.path.join(folder, exe)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


class FindToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.realpath(self._tmp.name)
        self._cwd = os.getcwd()
        self.addCleanup(os.chdir, self._cwd)

    def _dir(self, name):
        path = os.path.join(self.root, name)
        os.makedirs(path)
        return path

    def test_found_on_path(self):
        bin_dir = self._dir("bin")
        tool = _make_tool(bin_dir, "ffmpeg")
        with mock.patch.dict(os.environ, {"PATH": bin_dir}):
            self.assertEqual(media.find_tool("ffmpeg"), tool)

    def test_missing_returns_none(self):
        with mock.patch.dict(os.environ, {"PATH": self._dir("empty")}):
            self.assertIsNone(media.find_tool("ffmpeg"))

    def test_current_directory_is_never_searched(self):
        downloads = self._dir("downloads")
        _make_tool(downloads, "ffmpeg")
        os.chdir(downloads)
        # Empty and relative PATH entries both mean "the current directory"
        for path in ("", ".", os.pathsep.join(["", ".", self._dir("empty")])):
            with mock.patch.dict(os.environ, {"PATH": path}):
                self.assertIsNone(media.find_tool("ffmpeg"), path)

    def test_missing_tool_is_a_pipeline_error(self):
        from scrivox.core.pipeline import PipelineError
        with mock.patch.dict(os.environ, {"PATH": self._dir("empty")}):
            with self.assertRaises(PipelineError) as ctx:
                media.check_ffmpeg()
            self.assertIn("not found", str(ctx.exception))
            # The probes fall back instead of raising
            self.assertIsNone(media.get_media_duration("x.mp3"))
            self.assertEqual(media.list_audio_tracks("x.mp3"), [])
            self.assertTrue(media.has_video_stream("x.mp4"))
            self.assertFalse(media.has_video_stream("x.mp3"))


@unittest.skipUnless(media.find_tool("ffmpeg") and media.find_tool("ffprobe"),
                     "ffmpeg/ffprobe not installed")
class ProbeTests(unittest.TestCase):
    def test_file_name_starting_with_a_dash(self):
        """A relative name like '-clip.wav' must be read as a file, not as
        an ffprobe/ffmpeg option."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            self.addCleanup(os.chdir, cwd)
            os.chdir(tmp)
            subprocess.run(
                [media.find_tool("ffmpeg"), "-v", "error", "-f", "lavfi",
                 "-i", "anullsrc=r=16000:cl=mono", "-t", "1", "-y",
                 os.path.join(tmp, "-clip.wav")],
                check=True, timeout=60)
            self.assertAlmostEqual(media.get_media_duration("-clip.wav"), 1.0, places=1)
            self.assertEqual(len(media.list_audio_tracks("-clip.wav")), 1)
            self.assertFalse(media.has_video_stream("-clip.wav"))
            wav = media.extract_wav("-clip.wav", on_progress=lambda *_: None)
            try:
                self.assertGreater(os.path.getsize(wav), 1000)
            finally:
                os.remove(wav)


if __name__ == "__main__":
    unittest.main()
