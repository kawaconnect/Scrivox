#!/bin/sh
# Regenerates the hashed lock files the Windows release build installs from:
#   requirements-torch-cu126.lock  <- requirements-torch-cu126.in
#   requirements.lock              <- requirements.txt + requirements-build.in
# Needs uv (https://docs.astral.sh/uv/). Run from anywhere:
#   sh ci/update-locks.sh            keep locked versions where still allowed
#   sh ci/update-locks.sh --upgrade  move everything to the newest allowed
# The commands below are also recorded in each lock's header, which is what
# Renovate reads to refresh the locks; keep both in step. --universal (not
# --python-platform, which Renovate does not accept) makes one lock that is
# valid on Windows and Linux for Python 3.11, the build's Python.
set -eu
cd "$(dirname "$0")/.."
# Only torch and torchaudio come from the CUDA index (installed with
# --no-deps); their dependencies are locked from PyPI in requirements.lock,
# because the PyTorch index lists some files without hashes and the lock
# would then miss the hash of the Windows wheel.
uv pip compile requirements-torch-cu126.in --universal --python-version=3.11 \
  --generate-hashes --emit-index-url --index-url=https://download.pytorch.org/whl/cu126 \
  --no-emit-package=filelock --no-emit-package=fsspec --no-emit-package=jinja2 \
  --no-emit-package=markupsafe --no-emit-package=mpmath --no-emit-package=networkx \
  --no-emit-package=setuptools --no-emit-package=sympy --no-emit-package=typing-extensions \
  --output-file=requirements-torch-cu126.lock "$@"
# torch/torchaudio come from the CUDA lock above, never from PyPI.
uv pip compile requirements.txt requirements-build.in --universal --python-version=3.11 \
  --generate-hashes --no-emit-package=torch --no-emit-package=torchaudio \
  --output-file=requirements.lock "$@"
