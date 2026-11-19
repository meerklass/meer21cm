import pytest


def test_import():
    with pytest.raises(AttributeError):
        import meer21cm

        meer21cm.non_existent_module


def test_dir():
    import meer21cm

    assert "MockSimulation" in dir(meer21cm)


def test_dir_los():
    import meer21cm
    import meer21cm.los

    assert "LineOfSightPowerSpectrum" in dir(meer21cm)
    assert "RadialTransferFunction" in dir(meer21cm)
    assert meer21cm.LineOfSightPowerSpectrum is meer21cm.los.LineOfSightPowerSpectrum
