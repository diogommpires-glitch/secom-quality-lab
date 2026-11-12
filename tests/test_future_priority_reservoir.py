"""Exact future reservoir equivalence, including tie order and byte patterns."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "experiments/speed/priority_reservoir.py"
SPEC = importlib.util.spec_from_file_location("future_priority_reservoir", MODULE)
assert SPEC is not None and SPEC.loader is not None
MODULE_OBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE_OBJECT)
PriorityReservoir = MODULE_OBJECT.PriorityReservoir


def reference_update(previous, incoming, keys, capacity):
    values = incoming
    if previous is not None:
        values = np.concatenate((previous[0], values))
        keys = np.concatenate((previous[1], keys))
    if len(values) > capacity:
        keep = np.argpartition(keys, -capacity)[-capacity:]
        values, keys = values[keep], keys[keep]
    return values, keys


def assert_bytes(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("capacity", [1, 2, 7, 64])
@pytest.mark.parametrize("dimensions", [1, 3, 17])
def test_every_update_matches_original_bytes_order_and_rng(capacity, dimensions):
    rng, rng_reference = np.random.default_rng(20261026), np.random.default_rng(20261026)
    features_rng = np.random.default_rng(1234)
    candidate, reference = PriorityReservoir(capacity), None
    for rows in [0, 1, 8, 0, 3, 70, 5, 0, 20, 100]:
        features = features_rng.standard_normal((rows, dimensions), dtype=np.float32)
        keys, reference_keys = rng.random(rows), rng_reference.random(rows)
        reference = reference_update(reference, features, reference_keys, capacity)
        candidate.update(features, keys)
        assert_bytes(candidate.values(), reference[0])
        assert_bytes(candidate.priorities, reference[1])
        assert candidate.values().flags.c_contiguous
        assert rng.bit_generator.state == rng_reference.bit_generator.state


def test_ties_nonfinite_keys_and_feature_bit_patterns_preserve_exact_order():
    patterns = np.array(
        [0, 0x80000000, 0x7FC00001, 0x7FC00002, 0x7F800000, 0xFF800000], dtype=np.uint32
    )
    features = np.tile(patterns, 40).view(np.float32).reshape(60, 4)
    key_patterns = np.array([0.0, -0.0, 0.5, 0.5, np.inf, -np.inf, np.nan], dtype=np.float64)
    candidate, reference = PriorityReservoir(7), None
    for start, stop in [(0, 7), (7, 7), (7, 11), (11, 40), (40, 60)]:
        keys = np.resize(key_patterns, stop - start)
        values = features[start:stop]
        reference = reference_update(reference, values, keys, 7)
        candidate.update(values, keys)
        assert_bytes(candidate.values(), reference[0])
        assert_bytes(candidate.priorities, reference[1])


def test_strided_readonly_inputs_and_public_copies_cannot_mutate_storage():
    features = np.arange(120, dtype=np.float32).reshape(20, 6)[::2, ::2]
    keys = np.arange(20, dtype=np.float64)[::2]
    features.setflags(write=False)
    keys.setflags(write=False)
    reference = reference_update(None, features, keys, 5)
    candidate = PriorityReservoir(5)
    candidate.update(features, keys)
    values, priorities = candidate.values(), candidate.priorities
    values[:] = 999
    priorities[:] = -999
    assert_bytes(candidate.values(), reference[0])
    assert_bytes(candidate.priorities, reference[1])


@pytest.mark.parametrize(
    "features,keys",
    [
        (np.zeros((2, 3), dtype=np.float64), np.zeros(2, dtype=np.float64)),
        (np.zeros((2, 3), dtype=np.float32), np.zeros(2, dtype=np.float32)),
        (np.zeros((2, 3), dtype=np.float32), np.zeros(1, dtype=np.float64)),
        (np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float64)),
        (np.zeros((2, 0), dtype=np.float32), np.zeros(2, dtype=np.float64)),
        (np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float64)),
    ],
)
def test_invalid_update_leaves_valid_state_unchanged(features, keys):
    candidate = PriorityReservoir(2)
    candidate.update(np.ones((2, 3), dtype=np.float32), np.array([0.2, 0.8], dtype=np.float64))
    before = candidate.values(), candidate.priorities
    with pytest.raises(ValueError):
        candidate.update(features, keys)
    assert_bytes(candidate.values(), before[0])
    assert_bytes(candidate.priorities, before[1])


@pytest.mark.parametrize("capacity", [0, -1, 1.2, True, np.bool_(True), "3"])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        PriorityReservoir(capacity)


def test_full_reservoir_does_not_copy_retained_feature_rows():
    candidate = PriorityReservoir(4)
    candidate.update(np.ones((4, 10), dtype=np.float32), np.array([0.6, 0.7, 0.8, 0.9]))
    written = candidate.feature_rows_written
    candidate.update(np.zeros((100, 10), dtype=np.float32), np.linspace(0, 0.5, 100))
    assert candidate.feature_rows_written == written
    candidate.update(np.full((1, 10), 2, dtype=np.float32), np.array([1.0]))
    assert candidate.feature_rows_written == written + 1


def test_class_consumes_no_global_or_supplied_rng_state():
    rng = np.random.default_rng(111)
    keys = rng.random(8)
    state = rng.bit_generator.state.copy()
    legacy = np.random.get_state()
    candidate = PriorityReservoir(3)
    candidate.update(np.arange(16, dtype=np.float32).reshape(8, 2), keys)
    assert rng.bit_generator.state == state
    after = np.random.get_state()
    assert legacy[0] == after[0]
    assert np.array_equal(legacy[1], after[1])
    assert legacy[2:] == after[2:]


def test_copy_accounting_includes_incoming_gather_and_storage_assignment():
    candidate = PriorityReservoir(2)
    candidate.update(np.zeros((3, 4), dtype=np.float32), np.array([0.1, 0.8, 0.9]))
    assert candidate.feature_rows_written == 2
    assert candidate.feature_bytes_written == 2 * 2 * 4 * 4
    candidate.values()
    assert candidate.feature_bytes_written == (2 * 2 + 2) * 4 * 4
