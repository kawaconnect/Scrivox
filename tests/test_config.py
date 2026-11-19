"""Unit tests for the JSON config file (it holds the API keys).

Run with:  python -m unittest discover -s tests
"""

import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from scrivox import config as config_mod


class ConfigSaveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(config_mod, "_get_config_dir",
                                    return_value=self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_round_trip(self):
        cfg = config_mod.ConfigManager()
        cfg.set_credentials(hf_token="hf_x", openrouter_key="sk-x")
        cfg.save()
        again = config_mod.ConfigManager()
        self.assertEqual(again.get_credentials(), ("hf_x", "sk-x", ""))
        self.assertEqual(os.listdir(self._tmp.name), [config_mod.CONFIG_FILENAME])

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_file_is_private(self):
        cfg = config_mod.ConfigManager()
        cfg.save()
        self.assertEqual(stat.S_IMODE(os.stat(cfg.path).st_mode), 0o600)

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_existing_world_readable_file_becomes_private(self):
        path = os.path.join(self._tmp.name, config_mod.CONFIG_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"credentials": {"hf_token": "hf_old"}}, f)
        os.chmod(path, 0o644)
        cfg = config_mod.ConfigManager()
        cfg.save()
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(cfg.get_credentials()[0], "hf_old")

    def test_failed_save_keeps_the_old_file(self):
        cfg = config_mod.ConfigManager()
        cfg.set("ui", "geometry", "first")
        cfg.save()
        cfg.set("ui", "geometry", "second")
        with mock.patch.object(config_mod.json, "dump", side_effect=OSError("disk full")):
            cfg.save()
        self.assertEqual(config_mod.ConfigManager().get("ui", "geometry"), "first")
        self.assertEqual(os.listdir(self._tmp.name), [config_mod.CONFIG_FILENAME])


if __name__ == "__main__":
    unittest.main()
