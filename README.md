# Scrivox

GPU-accelerated transcription suite with speaker diarization, LLM translation, SRT subtitle generation, visual context analysis, and meeting summarization.

---

![Scrivox main window after a run: one file done, speaker identification and summary turned on, and the transcript with its summary on the right](docs/images/main-window.png)

## Features

- **Transcription**: faster-whisper on CUDA with float16 precision
- **Speaker Diarization**: identify who said what via pyannote
- **LLM Translation**: translate transcripts to one or many languages simultaneously (e.g. `ar,fr,ja`), producing a separate output file per language. Optionally translate summaries, vision descriptions, and document headers too.
- **SRT/VTT Subtitles**: subtitle files with optional speaker labels
- **Multi-Language**: auto-detect or specify primary language; per-segment language detection for mixed content (Korean+English, etc.)
- **Vision Analysis**: extract video keyframes and describe them with vision LLMs
- **Meeting Summary**: structured summaries with action items and key points
- **Multiple LLM Providers**: OpenRouter, OpenAI, Anthropic, Ollama, or any OpenAI-compatible endpoint
- **6 Output Formats**: txt, md, srt, vtt, json, tsv
- **GUI + CLI**: Tkinter desktop app or full command-line interface
- **Multi-Track Audio**: select specific audio tracks from multi-track video files
- **Batch Processing**: queue multiple files with drag-and-drop
- **Smart Caching**: transcription and diarization results cached per-file; cache auto-invalidates when model, language, or diarization params change
- **Portable Config**: JSON config stored next to the executable (or in `%APPDATA%\Scrivox` when that folder is read-only, as after the installer)

## Quick Start

### GUI

Double-click `Scrivox.exe` or run without arguments:

```
python main.py
```

The window walks you through three steps:

1. **Files**: drop audio or video files onto the window, or click to browse (Ctrl+O).
2. **Options**: pick the model and language. **Extras** (speaker names, on-screen
   content, summary, translation) fold out when you need them.
3. **Save to**: pick the format and folder. The transcript is always saved as a file,
   next to the original by default (e.g. `interview_transcript.txt`), and existing
   files are never overwritten. For a single file, **Rename…** sets the output name
   for that run (the GUI's replacement for the old free-text output path; the CLI's
   `--output` is unchanged).

Then press **Start transcription** (Ctrl+Enter). If something is missing (ffmpeg, a
graphics card, an API key), the window says so and offers **How to fix**. After
`winget install` of ffmpeg, **Check again** finds it without a restart. API keys,
subtitle timing and hardware options are under **Tools > Settings** (Ctrl+,).

### CLI

```bash
# Basic transcription
python main.py meeting.mp3

# Diarized SRT subtitles
python main.py video.mp4 --diarize --format srt -o subtitles.srt

# Full pipeline: diarize + vision + summary
python main.py meeting.mp4 --all --format md -o minutes.md

# Custom speaker names
python main.py meeting.mp4 --diarize --speaker-names "Alice,Bob,Charlie"

# Translate transcript to Arabic
python main.py meeting.mp3 --translate-to ar --format srt

# Translate to multiple languages at once (produces file.ar.srt, file.fr.srt, file.ja.srt)
python main.py meeting.mp3 --translate-to ar,fr,ja --format srt

# Translate everything: transcript + summary + vision + headers
python main.py meeting.mp4 --all --translate-to fr --translate-all --format md

# Korean drama with mixed language detection
python main.py kdrama.mp4 --language ko --format srt

# Use Anthropic Claude for translation/summary
python main.py meeting.mp3 --summarize --anthropic-key sk-ant-...

# Use OpenAI instead of OpenRouter
python main.py meeting.mp3 --summarize --api-base https://api.openai.com/v1/chat/completions --api-key sk-...

# Use local Ollama
python main.py meeting.mp3 --summarize --api-base http://localhost:11434/v1/chat/completions

# All features with JSON output
python main.py video.mp4 --all -f json -o report.json
```

### Dictation (Real-Time)

Source only (not part of the release builds):

```bash
python dictate.py                 # or dictate.bat
# Hold Ctrl+Shift to record, release to transcribe and type
# Ctrl+Shift+Q to quit

python dictate.py --list-devices  # microphones and their IDs
python dictate.py --device-id 3 --model small --language en
python dictate.py --copy-mode     # copy to the clipboard instead of typing
python dictate.py --hotkey ctrl+alt
```

The text is typed into whichever window has focus when you release the hotkey.

## Installation

### From Release (Recommended)

Download the latest release from [Releases](https://github.com/kawaconnect/Scrivox/releases).
Each variant comes as an installer and as a portable archive:

- **Installer** (`Scrivox*-win64-setup.exe`): installs to Program Files, adds
  Start menu shortcuts and an uninstaller. Enter API keys in the app under
  **Tools > Settings > AI services**.
- **Portable** (`Scrivox*-win64.7z`): extract anywhere with [7-Zip](https://www.7-zip.org/)
  and run `Scrivox.exe`. Keys can go in the app or in a `.env` file next to
  `Scrivox.exe` (copy `.env.example`).

A variant whose installer would be 2 GiB or larger ships only the `.7z`.
Each release has `SHA256SUMS-Scrivox*.txt` files to check the downloads
against (`Get-FileHash <file>` in PowerShell). The builds are not code signed,
so Windows SmartScreen may ask for confirmation on first run.

Three variants are available:

| Variant | Size | Description |
|---------|------|-------------|
| **Scrivox-Lite** | ~200-300 MB | Transcription only: no diarization, vision, summary or translation |
| **Scrivox** (Regular) | ~500-800 MB | All features; diarization models download on first use with your own HuggingFace token |
| **Scrivox-Full** | ~1.5-2 GB | All features with diarization models pre-bundled |

ffmpeg is not bundled: install it separately (see [Requirements](#requirements)).
The first run with a Whisper model downloads it from Hugging Face (75 MB for
`tiny` up to about 3 GB for `large-v3`) into the Hugging Face cache in your
user profile.

### From Source

Python 3.11 is what the releases are built and tested with. Install the CUDA
build of PyTorch first; a plain `pip install -r requirements.txt` would pull
the CPU-only build, and Scrivox would then report that no GPU was found.

```bash
git clone https://github.com/kawaconnect/Scrivox.git
cd Scrivox
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python main.py
```

Linux works from source for the CLI and GUI with an NVIDIA GPU; the releases
and the dictation tool are Windows only.

### Custom Models

You can provide your own models by placing them in a `models/` directory next to the executable:

```
Scrivox/
  Scrivox.exe
  models/
    whisper/
      large-v3/          # Custom Whisper model (CTranslate2 format)
    hub/                  # Custom HuggingFace models (diarization)
```

## Requirements

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support and a current driver. There is no CPU
  mode. The release builds bundle the CUDA libraries, so no CUDA toolkit
  install is needed.
- **ffmpeg and ffprobe** in PATH: `winget install --id Gyan.FFmpeg -e`, or a
  [manual download](https://ffmpeg.org/download.html). Scrivox also accepts
  `ffmpeg.exe` and `ffprobe.exe` placed next to `Scrivox.exe`. It does not
  run an ffmpeg found in the current directory.
- Internet access for the first-run model downloads and for the optional AI
  features (vision, summary, translation). Transcription itself runs locally.

## API Keys

Add to `.env` file or enter them in the GUI under **Tools > Settings > AI services**:

| Key | Required For | Get One |
|-----|-------------|---------|
| `HF_TOKEN` | Speaker diarization (Regular variant) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENROUTER_API_KEY` | Vision, summary, translation (if using OpenRouter) | [openrouter.ai/keys](https://openrouter.ai/keys) |

> **Note:** The Full variant has diarization models bundled, no HuggingFace token needed.
> Vision, summary, and translation features work with any OpenAI-compatible API provider or Anthropic.

For diarization with the Regular variant (or from source), you must also accept the model terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) with the account the token belongs to. The Lite variant has no diarization.

## Configuration

| What | Where |
|------|-------|
| GUI settings and API keys | `scrivox_config.json` next to `Scrivox.exe` (portable) or the repo root (from source). If that folder is not writable, as in an installed copy under Program Files, `%APPDATA%\Scrivox\scrivox_config.json`. The **AI services** settings tab shows the exact path. |
| `.env` | Next to `Scrivox.exe` or in the repo root. Read by the GUI and the CLI. |
| Error log | `scrivox_error.log` in the same folder as `scrivox_config.json` |
| Transcription cache | `<input file>.whisper_cache.json` next to each input file. Delete it or pass `--clear-cache` to transcribe again. |
| Whisper models | The Hugging Face cache (`%USERPROFILE%\.cache\huggingface`), or `models\whisper\<name>` next to `Scrivox.exe` if you supply your own |

Environment variables (in `.env` or the real environment):

| Variable | Used by | Meaning |
|----------|---------|---------|
| `HF_TOKEN` | GUI, CLI | Hugging Face token for diarization |
| `OPENROUTER_API_KEY` | GUI, CLI | OpenRouter API key for vision, summary and translation. Used only when the provider is OpenRouter (the CLI default); it is never sent to another server. For other providers enter the key in the GUI or pass `--api-key`. |
| `ANTHROPIC_API_KEY` | GUI only | Prefills the Anthropic key field. On the CLI use `--anthropic-key`. |
| `SCRIVOX_USE_SYSTEM_CUDA` | Release builds | `1` uses the CUDA libraries installed on the PC instead of the bundled ones (same as **Tools > Settings > Advanced**). |

The CLI does not read `scrivox_config.json` unless you pass `--use-config`.

## Security and privacy

- Transcription and diarization run on your PC. Vision, summary and
  translation send keyframes and transcript text to the AI provider you
  selected.
- API keys are stored unencrypted in `scrivox_config.json` and `.env`. Keep
  the Scrivox folder out of shared or synced locations, and prefer the
  installer (config under your own `%APPDATA%`) on a shared PC. Keys passed
  with `--api-key`, `--anthropic-key` or `--hf-token` are visible to other
  processes of the same user and may end up in shell history; `.env` avoids
  that.
- Each AI provider keeps its own saved key: switching the provider shows
  (and sends) only the key saved for that provider. A key saved by an older
  version, when one field served OpenRouter, OpenAI and Custom, is moved to
  the provider that was selected at the time (OpenRouter if Anthropic was).
  With `--use-config` the CLI likewise only uses the key saved for the server
  it talks to. Use an `https://` URL for any custom server that is not on
  this PC.
- Diarization models are PyTorch checkpoints, and pyannote loads them in a
  way that can run code from the file. The default model
  (`pyannote/speaker-diarization-community-1`) is downloaded, and bundled in
  Full, at a fixed commit (`DEFAULT_DIARIZATION_REVISION` in
  `scrivox/core/constants.py`) and loaded with PyTorch's `weights_only`
  mode, which only allows tensors and the three pyannote classes the
  checkpoints contain. Any other model ID or folder (Settings > Advanced >
  Speaker model, or `--diarization-model`) is loaded with code execution
  allowed: Scrivox warns in the settings and in the log, and you should only
  use models you trust.

## Troubleshooting

| Problem | What to do |
|---------|------------|
| "ffmpeg isn't installed" / `'ffmpeg' not found` | Install ffmpeg (see Requirements). In the GUI click **How to fix**, then **Check again**. A terminal opened before the install has the old PATH: open a new one. |
| "No NVIDIA graphics card found" / `CUDA GPU not available` | Update the NVIDIA driver. On a laptop, plug in the charger and set Scrivox to "High performance" in Windows graphics settings. From source, check that the CUDA build of torch is installed: `python -c "import torch; print(torch.cuda.is_available())"`. |
| Diarization fails with 401/403 or "gated" | The token is wrong, or the account has not accepted the pyannote model terms (see API Keys). Use **Test keys** in the settings. |
| "The graphics card ran out of memory" | Pick a smaller model (`large-v3-turbo`, `medium`, `small`) and close other GPU-heavy programs. |
| The first run seems stuck | It is downloading the Whisper model (up to about 3 GB). The GUI shows the progress; the CLI shows little until it finishes. |
| An old result comes back after changing something | Pass `--clear-cache` or delete `<input>.whisper_cache.json`. |
| CLI prints nothing from `Scrivox.exe` | Run it from an existing cmd or PowerShell window; the exe has no console of its own. |
| Something else | The **Log** tab has the details; unexpected errors are also written to `scrivox_error.log`. |

## LLM Providers

Scrivox supports multiple API providers for vision, summary, and translation features:

| Provider | Setup |
|----------|-------|
| **OpenRouter** (default) | Set `OPENROUTER_API_KEY` in `.env` |
| **OpenAI** | Use `--api-base https://api.openai.com/v1/chat/completions --api-key sk-...` |
| **Anthropic** | Use `--anthropic-key sk-ant-...` (auto-sets the Anthropic Messages API base) |
| **Ollama** (local) | Run Ollama locally, use `--api-base http://localhost:11434/v1/chat/completions` |
| **Custom** | Any endpoint that accepts the OpenAI chat completions format |

In the GUI, select your provider under **Tools > Settings > AI services**.

## CLI Reference

```
python main.py <input> [options]

Options:
  --model MODEL              Whisper model: tiny, base, small, medium, large-v3,
                             large-v3-turbo, distil-large-v3.5
  --language LANG            Primary language code (en, ko, ja, etc.) or auto-detect
  --format FORMAT            Output: txt, md, srt, vtt, json, tsv
  --output PATH, -o PATH     Output file path

Speaker Diarization:
  --diarize                  Enable speaker diarization
  --num-speakers N           Exact speaker count (if known)
  --min-speakers N           Minimum speakers expected
  --max-speakers N           Maximum speakers expected
  --speaker-names NAMES      Comma-separated names: "Alice,Bob"
  --diarization-model MODEL  Diarization model ID or local path

Vision Analysis:
  --vision                   Analyze video keyframes with vision LLM
  --vision-interval SEC      Seconds between keyframes (default: 60)
  --vision-model MODEL       Vision model (default: google/gemini-2.5-flash)
  --vision-workers N         Concurrent vision API requests (default: 4)
  --vision-change-threshold N  Skip near-duplicate frames (dhash bits, 0-64;
                             0 disables)

Meeting Summary:
  --summarize                Generate meeting summary
  --summary-model MODEL      Summary model (default: google/gemini-2.5-flash)

Translation:
  --translate-to LANGS       Target language(s), comma-separated: "ar" or "ar,fr,ja"
  --translate-all            Also translate summary, vision, and headers
  --translation-model MODEL  Translation model (default: google/gemini-2.5-flash)

Subtitle Tuning:
  --subtitle-speakers        Show speaker labels in SRT/VTT (off by default)
  --subtitle-max-chars N     Max characters per subtitle cue (default: 84)
  --subtitle-max-duration S  Max seconds per subtitle cue (default: 4.0)
  --subtitle-max-gap S       Max gap in seconds to merge across (default: 0.8)
  --subtitle-min-chars N     Min characters per cue when splitting (default: 15)

API & Credentials:
  --api-base URL             LLM API endpoint (default: OpenRouter)
  --api-key KEY              LLM API key for vision/summary/translation
  --anthropic-key KEY        Anthropic API key (auto-sets Anthropic base URL)
  --openrouter-key KEY       Older name for --api-key
  --hf-token TOKEN           HuggingFace token for diarization

Audio Tracks:
  --list-tracks              List audio tracks in the input file
  --audio-track N            Select audio track index (default: 0)

Other:
  --all                      Enable diarize + vision + summarize
  --use-config               Apply the settings saved by the Scrivox GUI as
                             defaults for this run (explicit flags still win)
  --no-diarize / --no-vision / --no-summarize / --no-translate
                             Turn a feature OFF even if --use-config/--all
                             enabled it
  --clear-cache              Force re-transcription
  --confidence-threshold F   Min avg word probability to keep a segment (default: 0.50)
```

### Scripting with your saved settings

`--use-config` makes the CLI honor whatever you configured in the GUI (model,
language, diarization, vision, output format, API keys/provider) without
retyping it:

```bash
python main.py meeting.mp4 --use-config
python main.py meeting.mp4 --use-config --format srt   # one override on top
```

Settings saved in the GUI that this build can't do (e.g. diarization on Lite)
are skipped with a notice instead of failing the run. This is also the hook
other apps use to drive Scrivox: [SimpleReliableRecorder](https://github.com/kawaconnect/SimpleReliableRecorder)
auto-detects a Scrivox install next to it and adds a "Transcribe with Scrivox"
button to its recordings library - transcripts follow your Scrivox settings.

## Building from Source

```bash
# Install the exact, hash-checked versions the release builds use
# (CUDA torch first; requirements.lock includes PyInstaller)
pip install --require-hashes --no-deps -r requirements-torch-cu126.lock
pip install --require-hashes -r requirements.lock

# Build all three variants: Lite, Regular and Full (Full requires HF_TOKEN env var)
python build.py --clean

# Build only Lite variant (no models needed)
python build.py --lite

# Build only Regular variant (all features, no bundled models)
python build.py --regular

# Build only Full variant (downloads diarization models)
set HF_TOKEN=hf_your_token_here
python build.py --full
```

## Building / Releases (GitLab CI)

Releases are built by GitLab CI (`.gitlab-ci.yml`). Pipelines run for `v*` tags, when started by hand
(web UI or API) and on merge requests, never on ordinary pushes. Merge request
pipelines run the tests; the Windows builds are manual there and nothing is
published.

| Stage | Job | Runner | What it does |
|-------|-----|--------|--------------|
| test | `test` | Linux (`python:3.11-bookworm`, Tk 8.6) | Unit and GUI tests under Xvfb with CPU-only torch. On release runs it also checks that `scrivox/__init__.py` has the release's version and that a `RELEASE_VERSION` run was started on that tag. |
| build | `build-windows-lite`, `build-windows-regular`, `build-windows-full` | Windows laptop (PowerShell shell runner) | Python 3.11, locked dependencies with CUDA 12.6 torch (see below), `build.py --clean --<variant>`, bundle checks, the portable `.7z` and the Inno Setup installer (`installer/windows.iss`). |
| release | `release` | Linux | Creates or updates the GitLab Release for the tag, linking the uploaded files. |

- **Windows toolchain:** `ci/tools-windows.ps1` installs Python 3.11.9 (NuGet
  package plus Tcl/Tk from python.org's `tcltk.msi`), Inno Setup 6.7.3 and
  7-Zip's `7zr.exe` once into `%ProgramData%\gitlab-runner-tools`, with every
  download pinned by SHA-256. `ci/build-windows.ps1` does the build and
  `ci/check_build.py` checks it: CUDA torch with its DLLs, `Scrivox.exe`,
  `.env.example`, Tk, no bundled `nvcuda.dll`, no pyannote in Lite, and the
  bundled models in Full.
- **Locked dependencies:** the Windows jobs install only from hashed lock
  files, with `pip install --require-hashes`: first
  `requirements-torch-cu126.lock` (only torch and torchaudio, from the CUDA
  12.6 index, installed with `--no-deps`), then `requirements.lock` (everything in `requirements.txt` plus
  PyInstaller from `requirements-build.in` and torch's own dependencies,
  without torch itself). The loose ranges
  stay in `requirements.txt`, `requirements-build.in` and
  `requirements-torch-cu126.in`; after changing one, regenerate the locks
  with [uv](https://docs.astral.sh/uv/): `sh ci/update-locks.sh` (add
  `--upgrade` to move every package to the newest allowed version). The
  locks are universal for Python 3.11 (valid on Windows and Linux). Renovate
  keeps them current through its `pip-compile` manager (`renovate.json`),
  which reruns the command recorded in each lock's header. The Linux test
  job still installs from `requirements.txt`, so it also catches breakage
  from new versions inside the allowed ranges.
- **One variant per job:** the laptop runs one job at a time, so the three
  variants build one after another. Each job may take up to 3 hours.
- **Full needs `HF_TOKEN`:** the Full variant downloads the gated pyannote
  diarization models, so its job only runs when an `HF_TOKEN` CI/CD variable
  is available to the pipeline. If the variable is protected, the `v*` tags
  must be protected too. Without `HF_TOKEN`, the pipeline builds Lite and
  Regular, and the release notes say that Full was not built.
- **Sizes:** each variant produces a `.7z` and a `-setup.exe` of roughly
  0.2-2 GB each, because of the CUDA DLLs and, in Full, the bundled models.
  That is far over GitLab's default 100 MB job-artifact limit, so the builds
  are not job artifacts. On release runs, each Windows job uploads its files
  straight to the project's Generic Package Registry (`Scrivox/<version>`;
  this GitLab instance sets no per-file limit there). The release links to
  those files. An untagged manual run builds and checks everything but keeps
  only the `SHA256SUMS-*.txt` files.
- **2 GiB limit:** GitLab releases are copied to GitHub by the release
  mirror, and GitHub rejects release assets of 2 GiB or more, so that is the
  limit that counts. An installer at or over 2 GiB is
  not published: the job warns, moves it to `out\oversize\` and leaves it
  out of `SHA256SUMS-<variant>.txt`, and the variant ships only its `.7z`
  portable. A `.7z` at or over 2 GiB fails the job. Full is the variant most
  likely to get there.

To release: bump `scrivox/__init__.py` and add a `CHANGELOG.md` entry for the
version (the `test` job checks both), then push the tag `vX.Y.Z`.

If a Windows job fails on a release run before it uploads anything, retry
that job; the variants that already uploaded are not touched. To rebuild an existing tag from scratch,
run a pipeline on the tag with `RELEASE_VERSION=vX.Y.Z`, but only while none
of that version's files have been published. The builds are not
reproducible, so the same file names would get new SHA-256 hashes, and the
release mirror fails for a tag whose already-copied asset changes its hash.
The Windows job therefore refuses to upload a file that is already in the
package version. Once any of a version's files are published, release a new
patch version instead.

## Tests

```bash
# Unit tests for output formatting and the LLM client (no GPU, network or API keys)
python -m unittest discover -s tests
# (a bare `python -m unittest` from the repo root also runs them;
#  test_setup.py is skipped there because it is a script, not a test module)

# GUI tests (tests/test_gui.py) need a display; on Linux run them under Xvfb.
# Run them with the Tk 8.6 that the .exe ships (python.org Python 3.11),
# not only Tk 9: widget sizes differ between the two.
python -c "import tkinter; print(tkinter.TkVersion)"   # expect 8.6
xvfb-run -a python -m unittest discover -s tests

# Environment check: CUDA, ffmpeg, audio devices, keys. Not offline: it
# downloads the large-v3 model (about 3 GB) if it is not cached, loads the
# diarization model when HF_TOKEN is set and calls the OpenRouter models API
# when OPENROUTER_API_KEY is set.
python test_setup.py
```

## Project Structure

```
scrivox/
  __init__.py              Package metadata
  cli.py                   CLI entry point (argparse -> pipeline)
  gui.py                   GUI entry point
  config.py                JSON config manager
  core/
    constants.py           Models, languages, extensions, LLM providers
    features.py            Runtime feature detection (Lite/Regular/Full)
    llm_client.py          Unified LLM API client (OpenAI + Anthropic formats)
    torch_compat.py        PyTorch 2.6+ compatibility
    media.py               ffmpeg utilities (extract, probe, duration)
    transcriber.py         Whisper transcription + post-processing
    diarizer.py            Speaker diarization (pyannote)
    translator.py          LLM-based segment/text/header translation
    vision.py              Keyframe extraction + vision LLM
    summarizer.py          Meeting summary generation
    formatter.py           Output formatting (6 formats)
    pipeline.py            Pipeline orchestrator (PipelineConfig -> run)
  ui/
    app.py                 Main application window (three-step flow)
    theme.py               Dark theme configuration
    widgets.py             Reusable widgets (drop zone, step cards, autocomplete combobox)
    log_redirect.py        Thread-safe stdout -> log widget
    output_paths.py        Output naming, accepted files, plain-language errors (no Tk)
    winnative.py           Windows niceties: DPI, dark title bar, taskbar flash, PATH refresh
    dialogs/
      track_dialog.py      Audio track selection dialog
      settings_dialog.py   Tools > Settings (AI services, advanced)
      help_dialog.py       "How to fix" help and the error report dialog
    frames/
      queue_frame.py       Job queue + multi-file browse + drag-and-drop
      settings_frame.py    Model, language, extras + their options
      models_frame.py      Advanced model and tuning settings
      api_frame.py         Provider selection + API key management
      output_frame.py      Format, save folder and output name
      progress_frame.py    Progress bar + elapsed timer
      log_frame.py         Scrollable log display (batched inserts)
      results_frame.py     Transcript display + copy/save
tests/                     Offline unit tests (unittest)
```

---

**KawaConnect LLC** | Built by Deenyoro
