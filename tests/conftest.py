import numpy as np
import pytest

from regimearb import Config
from regimearb.data import SyntheticSpec, make_universe


@pytest.fixture(scope="session")
def small_universe():
    spec = SyntheticSpec(n_days=1100, n_leaders=12, n_planted=5, seed=11)
    return make_universe(spec)


@pytest.fixture(scope="session")
def small_cfg():
    cfg = Config()
    cfg.walk_forward.formation = 300
    cfg.walk_forward.step = 100
    cfg.walk_forward.max_pairs = 6
    cfg.walk_forward.top_k = 30
    return cfg


@pytest.fixture
def rng():
    return np.random.default_rng(123)
