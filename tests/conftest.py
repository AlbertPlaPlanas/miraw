from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def original_weights() -> Path:
    """The original miRAW weights bundled with the package (always available)."""
    from pymiraw.storage import original_weights_path

    return original_weights_path()


@pytest.fixture(scope="session")
def paper_data() -> Path:
    """The unpacked paper archive (``pymiraw data unpack-archive PLOSComb.7z``); skipped when absent."""
    from pymiraw.storage import ORIGINAL_DIR

    if not (ORIGINAL_DIR / "NetworkModel" / "bestModel.bin").is_file():
        pytest.skip("paper data not unpacked: run `pymiraw data unpack-archive PLOSComb.7z`")
    return ORIGINAL_DIR


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "paper_data" in item.fixturenames:
            item.add_marker(pytest.mark.data)
