from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def original_weights() -> Path:
    """Original miRAW weights; the test is skipped when they cannot be obtained (no S3 access)."""
    from pymiraw.storage import original_weights_path

    try:
        return original_weights_path()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"original weights unavailable ({type(e).__name__}); set PYMIRAW_ORIGINAL_WEIGHTS")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "original_weights" in item.fixturenames:
            item.add_marker(pytest.mark.data)
