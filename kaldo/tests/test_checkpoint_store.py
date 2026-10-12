"""Block checkpoints are bound to their configuration and verified on reload."""

import numpy as np
import pytest

from kaldo.helpers.checkpoint_store import CheckpointStore


@pytest.fixture
def config():
    return {"format": "test-v1", "temperature_K": 300.0}


def test_saved_block_reloads_with_identical_arrays(tmp_path, config):
    store = CheckpointStore(tmp_path, config)
    values = np.arange(4, dtype=np.float64)
    store.save_block(0, 4, conductivity=values, evaluated_pairs=np.int64(16))
    loaded = CheckpointStore(tmp_path, config).load_block(0, 4, shapes={"conductivity": (4,)})
    np.testing.assert_array_equal(loaded["conductivity"], values)
    assert int(loaded["evaluated_pairs"]) == 16
    assert store.load_block(4, 8) is None


def test_changed_configuration_is_rejected(tmp_path, config):
    CheckpointStore(tmp_path, config).write_manifest(complete=False)
    with pytest.raises(ValueError, match="checkpoint configuration changed"):
        CheckpointStore(tmp_path, {**config, "temperature_K": 301.0})


def test_block_from_another_configuration_is_rejected(tmp_path, config):
    CheckpointStore(tmp_path, config).save_block(0, 2, conductivity=np.ones(2))
    with pytest.raises(ValueError, match="invalid checkpoint.*configuration or bounds"):
        CheckpointStore(tmp_path, {**config, "temperature_K": 301.0}).load_block(0, 2)


@pytest.mark.parametrize("damage", ["truncate", "shape"])
def test_damaged_block_is_rejected(tmp_path, config, damage):
    store = CheckpointStore(tmp_path, config)
    store.save_block(0, 2, conductivity=np.ones(2))
    if damage == "truncate":
        store.block_path(0, 2).write_bytes(b"damaged")
        message = "invalid checkpoint"
    else:
        message = "wrong shape"
    with pytest.raises(ValueError, match=message):
        store.load_block(0, 2, shapes={"conductivity": (3,)})
