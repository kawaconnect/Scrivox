"""Shared constants for the Scrivox transcription pipeline."""

from collections import OrderedDict

DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
DEFAULT_SUMMARY_MODEL = "google/gemini-2.5-flash"
DEFAULT_TRANSLATION_MODEL = "google/gemini-2.5-flash"

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
# Hugging Face commit of DEFAULT_DIARIZATION_MODEL that Scrivox downloads and
# bundles. Pinned so a changed upstream repo cannot swap the checkpoints
# under existing installs. To move it: pick the new commit on
# huggingface.co, check its checkpoints still load with
# scrivox.core.torch_compat (weights_only=True, see PYANNOTE_SAFE_GLOBALS)
# and update this value. build.py and ci/check_build.py read it from here.
DEFAULT_DIARIZATION_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
# Older saved settings name the pyannote 3.x pipeline; the diarizer
# replaces it with DEFAULT_DIARIZATION_MODEL.
LEGACY_DIARIZATION_MODELS = ("pyannote/speaker-diarization-3.1",)


def is_custom_diarization_model(model_id):
    """True when model_id is neither empty, the default model nor a legacy
    ID that is upgraded to it. Custom models are loaded with pickle code
    execution allowed (see scrivox.core.torch_compat)."""
    model_id = (model_id or "").strip()
    return bool(model_id) and model_id != DEFAULT_DIARIZATION_MODEL \
        and model_id not in LEGACY_DIARIZATION_MODELS


CUSTOM_DIARIZATION_MODEL_WARNING = (
    "Custom speaker models are loaded with code execution allowed: a malicious "
    "model can run any program on this computer. Only use models you trust.")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3.5"]
OUTPUT_FORMATS = ["txt", "md", "srt", "vtt", "json", "tsv"]

# LLM API providers
# OpenAI-compatible providers use /v1/chat/completions format.
# Anthropic uses its own Messages API format (handled by llm_client).
LLM_PROVIDERS = {
    "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
    "OpenAI": "https://api.openai.com/v1/chat/completions",
    "Anthropic": "https://api.anthropic.com/v1/messages",
    "Ollama (local)": "http://localhost:11434/v1/chat/completions",
}
DEFAULT_LLM_PROVIDER = "OpenRouter"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Preset model lists for combobox dropdowns
LLM_MODEL_PRESETS = [
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5",
    "openai/gpt-4.1-mini",
    "x-ai/grok-4",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3.5-plus",
    "mistralai/mistral-large-2512",
    "google/gemini-2.0-flash-exp:free",
]

VISION_MODEL_PRESETS = [
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5",
    "openai/gpt-4.1-mini",
    "x-ai/grok-4",
    "google/gemini-2.0-flash-exp:free",
]

# Whisper-supported languages: display name -> ISO 639-1 code
# Ordered alphabetically by display name for UI dropdowns
WHISPER_LANGUAGES = OrderedDict([
    ("Afrikaans", "af"),
    ("Arabic", "ar"),
    ("Armenian", "hy"),
    ("Azerbaijani", "az"),
    ("Belarusian", "be"),
    ("Bosnian", "bs"),
    ("Bulgarian", "bg"),
    ("Catalan", "ca"),
    ("Chinese", "zh"),
    ("Croatian", "hr"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("English", "en"),
    ("Estonian", "et"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("Galician", "gl"),
    ("German", "de"),
    ("Greek", "el"),
    ("Hebrew", "he"),
    ("Hindi", "hi"),
    ("Hungarian", "hu"),
    ("Icelandic", "is"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Kannada", "kn"),
    ("Kazakh", "kk"),
    ("Korean", "ko"),
    ("Latvian", "lv"),
    ("Lithuanian", "lt"),
    ("Macedonian", "mk"),
    ("Malay", "ms"),
    ("Marathi", "mr"),
    ("Maori", "mi"),
    ("Nepali", "ne"),
    ("Norwegian", "no"),
    ("Persian", "fa"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Romanian", "ro"),
    ("Russian", "ru"),
    ("Serbian", "sr"),
    ("Slovak", "sk"),
    ("Slovenian", "sl"),
    ("Spanish", "es"),
    ("Swahili", "sw"),
    ("Swedish", "sv"),
    ("Tagalog", "tl"),
    ("Tamil", "ta"),
    ("Thai", "th"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
    ("Urdu", "ur"),
    ("Vietnamese", "vi"),
    ("Welsh", "cy"),
])

# Reverse lookup: code -> display name
LANGUAGE_CODE_TO_NAME = {code: name for name, code in WHISPER_LANGUAGES.items()}

# Build translation target list: Whisper languages + regional variants
# inserted after their parent language for clean dropdown ordering
def _build_translation_languages():
    """Build ordered translation target list with regional variants."""
    # Variants to insert after their parent language
    _extras = {
        "zh": [("Chinese (Simplified)", "zh-CN"), ("Chinese (Traditional)", "zh-TW")],
        "pt": [("Portuguese (Brazil)", "pt-BR")],
    }
    items = []
    for name, code in WHISPER_LANGUAGES.items():
        items.append((name, code))
        if code in _extras:
            items.extend(_extras[code])
    return OrderedDict(items)

TRANSLATION_LANGUAGES = _build_translation_languages()
TRANSLATION_CODE_TO_NAME = {code: name for name, code in TRANSLATION_LANGUAGES.items()}
