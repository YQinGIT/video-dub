"""Stage 0 smoke test — the package imports and is versioned."""


def test_package_imports():
    import videodub

    assert videodub.__version__ == "0.1.0"
