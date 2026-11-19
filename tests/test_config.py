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


class ProviderKeyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(config_mod, "_get_config_dir",
                                    return_value=self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = os.path.join(self._tmp.name, config_mod.CONFIG_FILENAME)

    def write(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_legacy_key_moves_to_selected_provider(self):
        self.write({"credentials": {"openrouter_key": "sk-shared"},
                    "api": {"provider": "Custom", "custom_base": "https://llm.example"}})
        cfg = config_mod.ConfigManager()
        self.assertEqual(cfg.get_provider_key("Custom"), "sk-shared")
        self.assertEqual(cfg.get_provider_key("OpenRouter"), "")
        self.assertEqual(cfg.get_provider_key("OpenAI"), "")
        cfg.save()
        with open(self.path, encoding="utf-8") as f:
            saved = json.load(f)["credentials"]
        self.assertEqual(saved["openrouter_key"], "")
        self.assertEqual(saved["provider_keys"], {"Custom": "sk-shared"})

    def test_legacy_key_with_anthropic_selected_goes_to_openrouter(self):
        self.write({"credentials": {"openrouter_key": "sk-or", "anthropic_key": "sk-ant"},
                    "api": {"provider": "Anthropic"}})
        cfg = config_mod.ConfigManager()
        self.assertEqual(cfg.get_provider_key("OpenRouter"), "sk-or")
        self.assertEqual(cfg.get_provider_key("Anthropic"), "sk-ant")

    def test_legacy_key_without_provider_goes_to_openrouter(self):
        self.write({"credentials": {"openrouter_key": "sk-or"}})
        self.assertEqual(config_mod.ConfigManager().get_credentials()[1], "sk-or")

    def test_keys_are_kept_per_provider(self):
        cfg = config_mod.ConfigManager()
        cfg.set_credentials(provider_keys={"OpenAI": "sk-oa", "Custom": "sk-c", "OpenRouter": ""})
        cfg.save()
        again = config_mod.ConfigManager()
        self.assertEqual(again.get_provider_key("OpenAI"), "sk-oa")
        self.assertEqual(again.get_provider_key("Custom"), "sk-c")
        self.assertEqual(again.get_provider_key("OpenRouter"), "")
        self.assertEqual(again.get_provider_keys(), {"OpenAI": "sk-oa", "Custom": "sk-c"})


class CliSavedKeyTests(unittest.TestCase):
    """--use-config only uses the key saved for the server the run talks to."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(config_mod, "_get_config_dir",
                                    return_value=self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args_for(self, provider, argv=(), custom_base=""):
        from scrivox import cli
        cfg = config_mod.ConfigManager()
        cfg.set_credentials(provider_keys={"OpenRouter": "sk-or", "OpenAI": "sk-oa",
                                           "Custom": "sk-c"},
                            anthropic_key="sk-ant")
        cfg.set("api", "provider", provider)
        cfg.set("api", "custom_base", custom_base)
        cfg.save()
        args = cli.build_parser().parse_args(["in.wav", *argv])
        cli._apply_config_credentials(args)
        return args

    def test_each_provider_gets_its_own_key(self):
        self.assertEqual(self.args_for("OpenRouter").api_key, "sk-or")
        args = self.args_for("OpenAI")
        self.assertEqual((args.api_key, args.api_base),
                         ("sk-oa", "https://api.openai.com/v1/chat/completions"))
        args = self.args_for("Custom", custom_base="https://llm.example/v1/chat/completions")
        self.assertEqual(args.api_key, "sk-c")
        self.assertEqual(self.args_for("Anthropic").anthropic_key, "sk-ant")

    def test_custom_without_url_does_not_send_custom_key_to_openrouter(self):
        args = self.args_for("Custom")
        self.assertIsNone(args.api_base)
        self.assertEqual(args.api_key, "sk-or")

    def test_explicit_unknown_base_gets_no_saved_key(self):
        args = self.args_for("OpenRouter", ["--api-base", "https://other.example/v1"])
        self.assertIsNone(args.api_key)

    def test_explicit_key_wins(self):
        args = self.args_for("OpenAI", ["--api-key", "sk-explicit"])
        self.assertEqual(args.api_key, "sk-explicit")


if __name__ == "__main__":
    unittest.main()
