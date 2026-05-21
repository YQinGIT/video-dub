"""Smoke tests — the package imports cleanly with no CUDA installed."""


def test_package_imports():
    import videodub

    assert videodub.__version__ == "0.1.0"


def test_stage_packages_import_without_cuda():
    """Importing every swappable-backend stage must not require torch / CUDA.

    The factories live in each package's `__init__.py`; the heavy backends are
    imported lazily inside them. If any `__init__.py` pulled in `torch` at
    import time, this test would fail to collect — torch is not installed in
    the portable environment.
    """
    import videodub.asr
    import videodub.separation
    import videodub.timing
    import videodub.tts

    assert callable(videodub.asr.get_asr_backend)
    assert callable(videodub.separation.get_separator)
    assert callable(videodub.timing.get_timing_fitter)
    assert callable(videodub.tts.get_tts_backend)
