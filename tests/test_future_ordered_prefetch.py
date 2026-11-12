"""Future ordered-prefetch parity, resource bounds and deterministic training resume."""

import importlib.util
import json
import random
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "experiments/speed/ordered_prefetch.py"
SPEC = importlib.util.spec_from_file_location("future_ordered_prefetch", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ordered_prefetch = MODULE.ordered_prefetch


def producer_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith("secom-prefetch")]


@pytest.mark.parametrize("prefetch", [1, 2, 3])
@pytest.mark.parametrize("count", [0, 1, 7])
def test_ordered_bytes_and_single_producer(prefetch, count):
    owner = threading.get_ident()
    producer_ids = set()

    def prepare(index):
        producer_ids.add(threading.get_ident())
        return np.arange(20, dtype=np.float32).reshape(4, 5) + index

    actual = list(ordered_prefetch(range(count), prepare, prefetch))
    assert [value.tobytes() for value in actual] == [
        (np.arange(20, dtype=np.float32).reshape(4, 5) + index).tobytes() for index in range(count)
    ]
    assert len(producer_ids) == int(count > 0)
    assert owner not in producer_ids
    assert not producer_threads()


def test_prefetch_bound_includes_current_yielded_batch_and_early_close():
    prepared = []
    second = threading.Event()

    def prepare(index):
        prepared.append(index)
        if index == 1:
            second.set()
        return index

    iterator = ordered_prefetch(range(100), prepare, prefetch=2)
    try:
        assert next(iterator) == 0
        assert second.wait(2)
        assert prepared == [0, 1]
    finally:
        iterator.close()
    assert prepared == [0, 1]
    assert not producer_threads()


def test_close_joins_running_preparation():
    started, release, finished = (threading.Event() for _ in range(3))

    def prepare(index):
        if index == 1:
            started.set()
            assert release.wait(3)
            finished.set()
        return index

    iterator = ordered_prefetch(range(20), prepare)
    try:
        assert next(iterator) == 0
        assert started.wait(2)
    finally:
        release.set()
        iterator.close()
    assert finished.is_set()
    assert not producer_threads()


def test_prepare_failure_propagates_and_joins_producer():
    def prepare(index):
        if index == 1:
            raise ValueError("invalid deterministic input")
        return index

    iterator = ordered_prefetch(range(20), prepare)
    assert next(iterator) == 0
    with pytest.raises(ValueError, match="invalid deterministic input"):
        next(iterator)
    assert not producer_threads()


def test_item_iterator_failure_cancels_and_joins():
    def items():
        yield 0
        raise ValueError("broken frozen order")

    with pytest.raises(ValueError, match="broken frozen order"):
        list(ordered_prefetch(items(), lambda item: item))
    assert not producer_threads()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_prefetch_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        list(ordered_prefetch([1], lambda value: value, value))
    assert not producer_threads()


def rng_state(torch):
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        torch.get_rng_state().clone(),
    )


def assert_rng_equal(first, second, torch):
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_helper_preserves_rng_and_consumer_draw_order():
    torch = pytest.importorskip("torch")

    def seed():
        random.seed(83)
        np.random.seed(91)
        torch.manual_seed(79)

    seed()
    before = rng_state(torch)
    assert list(ordered_prefetch(range(8), lambda value: value * 2)) == list(range(0, 16, 2))
    assert_rng_equal(before, rng_state(torch), torch)

    def consume(values):
        return [
            (value, random.random(), float(np.random.random()), torch.rand(2).tolist())
            for value in values
        ]

    seed()
    expected = consume(value * 2 for value in range(8))
    final = rng_state(torch)
    seed()
    assert consume(ordered_prefetch(range(8), lambda value: value * 2)) == expected
    assert_rng_equal(final, rng_state(torch), torch)


def scientific_module(prefetch):
    source = (ROOT / "src/secom_quality/vision/models.py").read_bytes()
    if prefetch:
        spec = importlib.util.spec_from_file_location(
            "prefetch_optimizer", ROOT / "scripts/runpod/optimize_source.py"
        )
        optimizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimizer)
        source = optimizer.patch_ssn_prefetch(source, HELPER.read_bytes())
    module = types.ModuleType("secom_quality.vision.future_prefetch_test")
    module.__package__ = "secom_quality.vision"
    exec(compile(source, "future_prefetch_models.py", "exec"), module.__dict__)
    return module


def tiny_predictor(torch):
    class TinyModel(torch.nn.Module):
        fh, fw = 2, 2

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)
            self.classifier = torch.nn.Linear(3, 1)
            self.draws = []
            self.thread_ids = []

        def get_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.02, momentum=0.9)
            return optimizer, torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[1], gamma=0.7
            )

        def forward(self, images, masks, labels):
            draw = (random.random(), float(np.random.random()), float(torch.rand(())))
            self.draws.append(draw)
            self.thread_ids.append(threading.get_ident())
            maps = self.conv(images)
            scores = self.classifier(images.mean((2, 3))).flatten()
            return (
                torch.cat((maps, maps + sum(draw) / 20)),
                torch.cat((scores, scores + sum(draw) / 30)),
                torch.cat((masks, masks)),
                torch.cat((labels, labels)),
            )

    return SimpleNamespace(
        network=TinyModel(),
        device="cpu",
        resolution=(2, 2),
        config={"config_id": "synthetic-prefetch-parity", "epochs": 2, "lr_multiplier": 1.0},
        training={},
    )


def training_data():
    import pandas as pd

    return pd.DataFrame(
        {
            "record_id": list(range(12)),
            "image_path": [f"synthetic-{i}" for i in range(12)],
            "label": [1, 1, *([0] * 10)],
        }
    )


def pure_sample(row, resolution, flip):
    assert resolution == (2, 2)
    image = np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 30 + float(row.record_id) / 100
    mask = np.zeros((2, 2), dtype=np.float32)
    if row.label:
        mask[0, 0] = 1
    valid = np.ones((2, 2), dtype=np.float32)
    weights = np.ones((2, 2), dtype=np.float32)
    if flip in {1, 3}:
        image, mask, valid, weights = (
            np.flip(image, 1),
            np.flip(mask, 0),
            np.flip(valid, 0),
            np.flip(weights, 0),
        )
    if flip in {2, 3}:
        image, mask, valid, weights = (
            np.flip(image, 2),
            np.flip(mask, 1),
            np.flip(valid, 1),
            np.flip(weights, 1),
        )
    return tuple(item.copy() for item in (image, mask, valid, weights))


def run_training(path, prefetch, *, interrupt=False, resume=False):
    torch = pytest.importorskip("torch")
    module = scientific_module(prefetch)
    module.seed_runtime(123 if not resume else 999)
    predictor = tiny_predictor(torch)
    module._training_sample = pure_sample
    original_save = module._save_checkpoint

    def save(*args):
        original_save(*args)
        if interrupt and args[4] == 1:
            raise InterruptedError("after durable epoch one")

    module._save_checkpoint = save
    if interrupt:
        with pytest.raises(InterruptedError, match="durable epoch one"):
            module._fit_ssn(predictor, training_data(), 123, path)
    else:
        module._fit_ssn(predictor, training_data(), 123, path)
    assert all(value == threading.get_ident() for value in predictor.network.thread_ids)
    assert not producer_threads()
    return predictor, rng_state(torch)


def assert_training_equal(first_path, second_path, first, second, first_rng, second_rng):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    for name, tensor in first.network.state_dict().items():
        assert torch.equal(tensor, second.network.state_dict()[name]), name
    for key in ("epoch", "epochs", "examples", "mean_loss"):
        assert [
            row[key] for row in json.loads((first_path / "training_history.json").read_text())
        ] == [row[key] for row in json.loads((second_path / "training_history.json").read_text())]
    assert_rng_equal(first_rng, second_rng, torch)
    first_metadata = json.loads((first_path / "checkpoint.json").read_text())
    second_metadata = json.loads((second_path / "checkpoint.json").read_text())
    assert first_metadata["state"] == second_metadata["state"]
    first_tensors, second_tensors = (
        load_file(first_path / "checkpoint.safetensors"),
        load_file(second_path / "checkpoint.safetensors"),
    )
    assert first_tensors.keys() == second_tensors.keys()
    assert all(torch.equal(tensor, second_tensors[name]) for name, tensor in first_tensors.items())


def test_complete_ssn_fit_preserves_weights_losses_rng_and_checkpoint(tmp_path):
    first, first_rng = run_training(tmp_path / "original", False)
    second, second_rng = run_training(tmp_path / "prefetched", True)
    assert first.network.draws == second.network.draws
    assert_training_equal(
        tmp_path / "original", tmp_path / "prefetched", first, second, first_rng, second_rng
    )


def test_prefetched_checkpoint_resume_matches_original_continuous(tmp_path):
    expected, expected_rng = run_training(tmp_path / "original", False)
    interrupted, _ = run_training(tmp_path / "resumed", True, interrupt=True)
    resumed, resumed_rng = run_training(tmp_path / "resumed", True, resume=True)
    assert expected.network.draws == interrupted.network.draws + resumed.network.draws
    assert_training_equal(
        tmp_path / "original", tmp_path / "resumed", expected, resumed, expected_rng, resumed_rng
    )


def test_training_exception_closes_and_joins_producer(tmp_path):
    torch = pytest.importorskip("torch")
    module = scientific_module(True)
    module.seed_runtime(123)
    predictor = tiny_predictor(torch)
    count = 0
    started, release, finished = (threading.Event() for _ in range(3))

    def prepare(row, resolution, flip):
        nonlocal count
        count += 1
        if count == 9:
            started.set()
            assert release.wait(3)
        result = pure_sample(row, resolution, flip)
        if count == 16:
            finished.set()
        return result

    def fail(*_args):
        assert started.wait(2)
        release.set()
        raise RuntimeError("model failure")

    module._training_sample = prepare
    predictor.network.forward = fail
    try:
        with pytest.raises(RuntimeError, match="model failure"):
            module._fit_ssn(predictor, training_data(), 123, tmp_path / "failure")
    finally:
        release.set()
    assert finished.is_set()
    assert not producer_threads()
    assert not (tmp_path / "failure/checkpoint.json").exists()
