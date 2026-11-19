"""Control how torch.load behaves while pyannote loads a diarization model.

pyannote.audio 4.x calls torch.load (through Lightning) with
weights_only=False for every model checkpoint, which unpickles arbitrary
objects and so lets a checkpoint run code on load. Scrivox narrows that:

* Trusted models (the default model at its pinned revision, or the copy
  bundled with the Full build) are loaded with weights_only=True. The few
  pyannote classes those checkpoints contain are registered with
  torch.serialization.add_safe_globals, so nothing else can be unpickled.
* Any other model ID or local folder is loaded the way pyannote does it,
  with code execution allowed. Callers must warn the user first.

Both modes patch torch.load only for the duration of the load. The patch
is reference-counted and both modes cannot overlap: a load in one mode
waits for loads in the other mode to finish, so a concurrent unsafe load
can never relax a trusted load (or the other way round).
"""

import contextlib
import threading

import torch

_original_torch_load = torch.load
_patch_lock = threading.Condition()
_patch_depth = 0
_patch_mode = None

# Fully qualified names of the non-torch globals found in the pinned
# pyannote/speaker-diarization-community-1 checkpoints (segmentation and
# embedding pytorch_model.bin). Everything else they contain (tensors,
# storages, OrderedDict, primitives) is already allowed by weights_only.
PYANNOTE_SAFE_GLOBALS = (
    "pyannote.audio.core.task.Problem",
    "pyannote.audio.core.task.Resolution",
    "pyannote.audio.core.task.Specifications",
)

_safe_globals_registered = False


def _register_safe_globals():
    """Allow the pyannote task classes in weights_only checkpoints (once)."""
    global _safe_globals_registered
    if _safe_globals_registered:
        return
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    torch.serialization.add_safe_globals([Problem, Resolution, Specifications])
    _safe_globals_registered = True


def _make_patched(weights_only):
    def patched(*args, **kwargs):
        kwargs["weights_only"] = weights_only
        return _original_torch_load(*args, **kwargs)
    return patched


@contextlib.contextmanager
def pyannote_torch_load(trusted):
    """Patch torch.load for a pyannote model load.

    trusted=True forces weights_only=True with only the pyannote task
    classes allowed (a checkpoint that needs anything else fails to load).
    trusted=False forces weights_only=False, which allows code execution.
    """
    global _patch_depth, _patch_mode
    mode = "trusted" if trusted else "unsafe"
    if trusted:
        _register_safe_globals()

    with _patch_lock:
        while _patch_depth and _patch_mode != mode:
            _patch_lock.wait()
        if _patch_depth == 0:
            torch.load = _make_patched(trusted)
            _patch_mode = mode
        _patch_depth += 1
    try:
        yield
    finally:
        with _patch_lock:
            _patch_depth -= 1
            if _patch_depth == 0:
                torch.load = _original_torch_load
                _patch_mode = None
                _patch_lock.notify_all()
