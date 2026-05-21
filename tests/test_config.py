"""Stage 1 — Settings / config tests."""

from videodub.config import Settings


def test_defaults():
    s = Settings()
    assert s.asr.backend == "faster_whisper"
    assert s.translation.backend == "deepseek"
    assert s.separation.backend == "demucs"
    assert s.separation.enabled is True
    assert s.tts.backend == "cosyvoice2"
    assert s.timing.backend == "rubberband"
    assert s.deepseek_api_key is None


def test_nested_env_override(monkeypatch):
    monkeypatch.setenv("VIDEODUB_ASR__BACKEND", "mock")
    monkeypatch.setenv("VIDEODUB_TRANSLATION__BACKEND", "mock")
    monkeypatch.setenv("VIDEODUB_SEPARATION__ENABLED", "false")
    s = Settings()
    assert s.asr.backend == "mock"
    assert s.translation.backend == "mock"
    assert s.separation.enabled is False


def test_secret_is_loaded_and_not_leaked(monkeypatch):
    monkeypatch.setenv("VIDEODUB_DEEPSEEK_API_KEY", "sk-test-secret-123")
    s = Settings()
    assert s.deepseek_api_key is not None
    assert s.deepseek_api_key.get_secret_value() == "sk-test-secret-123"
    # The raw secret must not appear in repr/str.
    assert "sk-test-secret-123" not in repr(s)
    assert "sk-test-secret-123" not in str(s.deepseek_api_key)
