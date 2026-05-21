"""Stage 1 — exception hierarchy tests."""

from videodub.errors import BackendError, ConfigError, StageError, VideodubError


def test_all_errors_subclass_base():
    for exc in (ConfigError, BackendError, StageError):
        assert issubclass(exc, VideodubError)
        assert issubclass(exc, Exception)
