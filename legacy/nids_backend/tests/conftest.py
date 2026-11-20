import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nids import Config, train  # noqa: E402
from nids.data import load_splits  # noqa: E402


@pytest.fixture(scope="session")
def splits():
    return load_splits(Config.fast())


@pytest.fixture(scope="session")
def bundle(splits):
    return train(Config.fast(), splits)
