import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def original_weights() -> Path:
    """Original miRAW weights; the test is skipped when they cannot be obtained (no S3 access)."""
    from pymiraw import storage
    from pymiraw.storage import original_weights_path

    has_local = os.environ.get(storage.ORIGINAL_WEIGHTS_ENV) or storage.LOCAL_WEIGHTS.is_file()
    has_cache = storage.local_path(storage.ORIGINAL_WEIGHTS_KEY).is_file()
    has_aws = any(k in os.environ for k in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_WEB_IDENTITY_TOKEN_FILE"))
    if not (has_local or has_cache or has_aws):  # fail fast: do not wait for S3 credential-lookup retries
        pytest.skip("original weights unavailable: run `pymiraw data unpack-archive` or set PYMIRAW_ORIGINAL_WEIGHTS")
    try:
        return original_weights_path()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"original weights unavailable ({type(e).__name__}); set PYMIRAW_ORIGINAL_WEIGHTS")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "original_weights" in item.fixturenames:
            item.add_marker(pytest.mark.data)
