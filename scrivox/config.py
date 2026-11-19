"""ConfigManager: JSON-based config persistence next to exe or in %APPDATA%."""

import json
import os
import sys


def _get_config_dir():
    """Determine config directory: next to exe (frozen) or next to project root (dev)."""
    if getattr(sys, "frozen", False):
        # PyInstaller exe, config lives next to the exe
        exe_dir = os.path.dirname(sys.executable)
        # Test if writable
        test_path = os.path.join(exe_dir, ".scrivox_write_test")
        try:
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
            return exe_dir
        except OSError:
            pass
    else:
        # Dev mode, config lives next to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return project_root

    # Fallback: %APPDATA%\Scrivox\
    appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
    config_dir = os.path.join(appdata, "Scrivox")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


CONFIG_FILENAME = "scrivox_config.json"

_DEFAULT_CONFIG = {
    "credentials": {
        "hf_token": "",
        # Legacy single key for every OpenAI-compatible provider; moved into
        # provider_keys on load (see _migrate_provider_keys) and kept empty.
        "openrouter_key": "",
        "anthropic_key": "",
        # One key per OpenAI-compatible provider (OpenRouter, OpenAI,
        # Ollama (local), Custom), so switching provider never sends one
        # provider's key to another provider's server.
        "provider_keys": {},
    },
    "last_settings": {
        "model": "large-v3",
        "language": "",
        "diarize": False,
        "vision": False,
        "summarize": False,
        "output_format": "txt",
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
        "speaker_names": "",
        "vision_interval": 60,
        "vision_model": "google/gemini-2.5-flash",
        "vision_workers": 4,
        "summary_model": "google/gemini-2.5-flash",
        "diarization_model": "pyannote/speaker-diarization-community-1",
        "use_system_cuda": False,
        "subtitle_max_chars": 84,
        "subtitle_max_duration": 4.0,
        "subtitle_max_gap": 0.8,
        "subtitle_min_chars": 15,
        "subtitle_speakers": False,
        "confidence_threshold": 0.50,
        "translate": False,
        "translate_all": False,
        "translate_to": "",
        "translation_model": "google/gemini-2.5-flash",
    },
    "api": {
        "provider": "OpenRouter",
        "custom_base": "",
    },
    "ui": {
        "geometry": "",
        "last_input_dir": "",
        "last_output_dir": "",
        "preferred_language": "",
        "recent_files": [],
    },
}


class ConfigManager:
    """Read/write JSON config file with section-based access."""

    def __init__(self):
        self._dir = _get_config_dir()
        self._path = os.path.join(self._dir, CONFIG_FILENAME)
        self._data = {}
        self._load()

    @property
    def path(self):
        return self._path

    def _load(self):
        """Load config from disk, merging with defaults."""
        self._data = json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                # Merge saved values into defaults (preserving new default keys)
                for section in _DEFAULT_CONFIG:
                    if section in saved and isinstance(saved[section], dict):
                        self._data[section].update(saved[section])
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                pass  # corrupt file, use defaults
        self._migrate_provider_keys()

    def _migrate_provider_keys(self):
        """Move the old shared API key to the provider it was used with.

        Before per-provider keys, one "openrouter_key" served OpenRouter,
        OpenAI, Ollama and Custom alike. It goes to the provider selected
        now (OpenRouter if Anthropic is selected, since Anthropic always had
        its own key). Takes effect on disk with the next save.
        """
        creds = self._data["credentials"]
        if not isinstance(creds.get("provider_keys"), dict):
            creds["provider_keys"] = {}
        legacy = (creds.get("openrouter_key") or "").strip()
        if not legacy:
            return
        provider = self.get("api", "provider", "OpenRouter") or "OpenRouter"
        if provider == "Anthropic":
            provider = "OpenRouter"
        creds["provider_keys"].setdefault(provider, legacy)
        creds["openrouter_key"] = ""

    def save(self):
        """Persist config to disk."""
        # The file holds API keys: write a private temp file (0600 where the
        # OS has file modes) and swap it in, so a crash mid-write can't leave
        # a truncated config and other local users can't read the keys.
        tmp_path = self._path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except OSError:
            # silently fail if can't write
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def get(self, section, key, default=None):
        """Get a config value."""
        return self._data.get(section, {}).get(key, default)

    def set(self, section, key, value):
        """Set a config value (does not auto-save)."""
        if section not in self._data:
            self._data[section] = {}
        self._data[section][key] = value

    def get_section(self, section):
        """Get an entire config section as a dict."""
        return dict(self._data.get(section, {}))

    def get_credentials(self):
        """Return (hf_token, openrouter_key, anthropic_key) from config.

        openrouter_key is the key saved for the OpenRouter provider only;
        use get_provider_key() for the others.
        """
        creds = self._data.get("credentials", {})
        return (creds.get("hf_token", ""),
                self.get_provider_key("OpenRouter"),
                creds.get("anthropic_key", ""))

    def get_provider_key(self, provider):
        """The API key saved for one LLM provider ("" if none)."""
        if provider == "Anthropic":
            return self._data.get("credentials", {}).get("anthropic_key", "")
        keys = self._data.get("credentials", {}).get("provider_keys") or {}
        return keys.get(provider, "") if isinstance(keys, dict) else ""

    def get_provider_keys(self):
        """Copy of the per-provider keys for OpenAI-compatible providers."""
        keys = self._data.get("credentials", {}).get("provider_keys") or {}
        return dict(keys) if isinstance(keys, dict) else {}

    def set_credentials(self, hf_token=None, openrouter_key=None, anthropic_key=None,
                        provider_keys=None):
        """Update credential values.

        openrouter_key sets the OpenRouter provider's key; provider_keys
        replaces the keys of all OpenAI-compatible providers at once.
        """
        if hf_token is not None:
            self.set("credentials", "hf_token", hf_token)
        if provider_keys is not None:
            self.set("credentials", "provider_keys",
                     {k: v for k, v in provider_keys.items() if v})
        if openrouter_key is not None:
            keys = self.get_provider_keys()
            keys["OpenRouter"] = openrouter_key
            self.set("credentials", "provider_keys", {k: v for k, v in keys.items() if v})
        if anthropic_key is not None:
            self.set("credentials", "anthropic_key", anthropic_key)

    def save_last_settings(self, **kwargs):
        """Save last-used pipeline settings."""
        for key, value in kwargs.items():
            self.set("last_settings", key, value)
        self.save()

    def get_last_settings(self):
        """Get last-used pipeline settings."""
        return self.get_section("last_settings")
