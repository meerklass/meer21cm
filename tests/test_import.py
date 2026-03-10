import pytest


def test_import():
    with pytest.raises(AttributeError):
        import meer21cm

        meer21cm.non_existent_module


def test_dir():
    import meer21cm

    assert "MockSimulation" in dir(meer21cm)
