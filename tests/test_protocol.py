"""Behavioral checks for held-out evaluation and operation accounting."""
import json
import math
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from snn_experiment.data import stratified_split, stratified_subset, make_loaders
from snn_experiment.energy import profile_operations
from snn_experiment.evaluation import evaluate, classification_report
from snn_experiment.hardware import integrate_power, DeviceMeter
from snn_experiment.models import CNN, SNN, LIFNeuron, SpikeFunction, TEBN, EveryTimestep
from snn_experiment.run import Config, run, seed_everything

torch.set_num_threads(2)


def test_stratified_holdout_is_disjoint_balanced_and_repeatable():
    targets = np.repeat(np.arange(10), 20)
    train, validation = stratified_split(targets, 3, 7, seed=19)
    assert not set(train) & set(validation)
    assert np.bincount(targets[train]).tolist() == [7]*10
    assert np.bincount(targets[validation]).tolist() == [3]*10
    assert (train, validation) == stratified_split(targets, 3, 7, seed=19)
    assert (train, validation) != stratified_split(targets, 3, 7, seed=20)
    with pytest.raises(ValueError):
        stratified_split(targets, 20)
    with pytest.raises(ValueError):
        stratified_subset(targets, 21, 42)


def test_validation_uses_separate_transform_and_original_training_indices(monkeypatch):
    import torchvision.datasets
    calls = []
    class FakeCIFAR(TensorDataset):
        def __init__(self, root, train, download, transform):
            self.targets = np.repeat(np.arange(10), 6).tolist()
            self.transform, self.train = transform, train
            calls.append(self)
            super().__init__(torch.zeros(60, 3, 32, 32), torch.tensor(self.targets))
    monkeypatch.setattr(torchvision.datasets, "CIFAR10", FakeCIFAR)
    loaders, manifest = make_loaders(Config(validation_per_class=2, train_per_class=3, test_per_class=2))
    assert calls[0].train and calls[1].train and not calls[2].train
    assert "RandomCrop" in repr(calls[0].transform)
    assert "Random" not in repr(calls[1].transform)
    assert repr(calls[1].transform) == repr(calls[2].transform)
    assert not set(manifest["train_indices"]) & set(manifest["validation_indices"])
    assert len(loaders["train"].dataset) == 30


def test_metrics_use_counts_and_handle_unpredicted_classes():
    result = classification_report([[8, 2], [1, 1]])
    assert result["samples"] == 12 and result["accuracy_percent"] == 75
    assert result["per_class"][1]["recall"] == 0.5
    lo, hi = result["accuracy_wilson95_percent"]
    assert lo < 75 < hi
    assert classification_report([[1, 0], [1, 0]])["per_class"][1]["precision"] == 0
    with pytest.raises(ValueError):
        classification_report(torch.zeros(10, 10))


def test_evaluation_weights_partial_batches_and_restores_mode():
    labels = torch.tensor([0, 0, 0, 0, 1])
    loader = DataLoader(TensorDataset(torch.zeros(5, 10), labels), batch_size=4)
    model = nn.Identity().train()
    result = evaluate(model, loader, "cpu")
    assert result["accuracy_percent"] == 80 and result["samples"] == 5
    assert result["loss"] == pytest.approx(math.log(10))
    assert model.training


def test_spike_threshold_surrogate_and_membrane_reset():
    x = torch.tensor([-1.5, -0.5, 0., 0.5, 1.5], requires_grad=True)
    y = SpikeFunction.apply(x)
    assert y.tolist() == [0, 0, 1, 1, 1]
    y.sum().backward()
    assert x.grad.tolist() == [0, 0.5, 1, 0.5, 0]
    neuron = LIFNeuron()
    assert neuron(torch.full((1, 3, 1), .75)).flatten().tolist() == [0, 1, 0]
    assert neuron(torch.full((1, 3, 1), .75)).flatten().tolist() == [0, 1, 0]


def test_tebn_joint_statistics_temporal_scale_and_gradient():
    norm = TEBN(2, 3)
    x = torch.randn(2, 3, 2, 4, 4, requires_grad=True)
    with torch.no_grad():
        norm.timestep_scale.copy_(torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1, 1))
    expected = nn.functional.batch_norm(x.flatten(0, 1), None, None, training=True).reshape_as(x)
    result = norm(x)
    assert torch.allclose(result, expected * norm.timestep_scale, atol=1e-6)
    result.square().sum().backward()
    assert norm.timestep_scale.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        norm(torch.randn(2, 4, 2, 4, 4))


@pytest.mark.parametrize("model,parameters", [(CNN, 2197706), (SNN, 2197730)])
def test_original_architectures_have_finite_backward_pass(model, parameters):
    network = model()
    assert sum(p.numel() for p in network.parameters()) == parameters
    logits = network(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


def test_operation_counts_include_groups_time_and_partial_batches():
    model = nn.Conv2d(2, 2, 1, groups=2, bias=False)
    loader = DataLoader(TensorDataset(torch.ones(5, 2, 2, 2), torch.zeros(5)), batch_size=3)
    result = profile_operations(model, loader, "cpu", max_batches=3)
    assert result["macs_per_image"] == 8
    timed = nn.Sequential(EveryTimestep(nn.Linear(3, 2, bias=False)))
    timed_loader = DataLoader(TensorDataset(torch.ones(5, 4, 3), torch.zeros(5)), batch_size=3)
    assert profile_operations(timed, timed_loader, "cpu")["macs_per_image"] == 24
    assert not result["layers"][0]["spike_input"]
    assert result["estimated_acs_per_image"] == 0


def test_profile_hooks_are_removed_after_failure():
    class Broken(nn.Sequential):
        def forward(self, x):
            super().forward(x)
            raise RuntimeError("broken")
    model = Broken(nn.Linear(2, 2)).train()
    loader = DataLoader(TensorDataset(torch.ones(1, 2), torch.zeros(1)))
    with pytest.raises(RuntimeError, match="broken"):
        profile_operations(model, loader, "cpu")
    assert not model[0]._forward_hooks and model.training


def test_energy_integrates_irregular_samples_and_cpu_is_missing():
    assert integrate_power([(0, 10), (1, 20), (3, 10)], 0, 3) == 45
    assert integrate_power([(1, 10)], 0, 3) is None
    with DeviceMeter("cpu") as meter:
        pass
    assert meter.result["energy_joules"] is None
    assert meter.result["peak_cuda_allocated_mb"] is None


def test_seed_reproduces_python_numpy_and_torch():
    import random
    seed_everything(42)
    first = random.random(), np.random.rand(), torch.rand(2)
    seed_everything(42)
    second = random.random(), np.random.rand(), torch.rand(2)
    assert first[:2] == second[:2] and torch.equal(first[2], second[2])


def test_zero_validation_accuracy_still_saves_checkpoint_and_test_is_held_out(tmp_path, monkeypatch):
    import snn_experiment.run as runner
    loaders, manifest = make_loaders(Config(smoke=True, epochs=1, batch_size=8))
    real_evaluate = runner.evaluate
    calls = []
    def evaluation(model, loader, device):
        calls.append(id(loader))
        result = real_evaluate(model, loader, device)
        result["accuracy_percent"] = 0.0
        return result
    monkeypatch.setattr(runner, "evaluate", evaluation)
    monkeypatch.setattr(runner, "make_loaders", lambda config: (loaders, manifest))
    output = tmp_path/"experiment"
    result = run(Config(model="cnn", smoke=True, epochs=1, batch_size=8, threads=2,
                        benchmark_repeats=1, output=str(output)))
    assert (output/"cnn/best.pt").exists() and result[0]["best_epoch"] == 1
    assert calls == [id(loaders["validation"]), id(loaders["test"])]
    metadata = json.loads((output/"run.json").read_text())
    assert metadata["status"] == "completed" and metadata["test_selection"] is False
    assert json.loads((output/"cnn/test_metrics.json").read_text())["samples"] == 20
    with pytest.raises(FileExistsError):
        run(Config(model="cnn", smoke=True, epochs=1, output=str(output)))


@pytest.mark.parametrize("name", ["CNN", "SNN"])
def test_extracted_models_match_original_notebook_outputs(name):
    from pathlib import Path
    import subprocess
    try:
        raw = subprocess.check_output(
            ["git", "show", "26ec7d26b32e02913c83c093911737a29e5807bb:notebooks/snn_vs_ann_tebn.ipynb"],
            cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("immutable baseline Git history is required for original-model parity")
    notebook = json.loads(raw)
    # Select definitions only, without data loading, training or notebook magics.
    namespace = {"torch": torch, "nn": nn, "TIMESTEPS": 4, "DECAY": 0.5, "THRESHOLD": 1.0}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith(("def conv_bn_relu", "class SpikeFunction", "class LIFNeuron", "class EveryTimestep", "class TEBN", "def conv_norm_lif")):
            exec(compile(source, "original_notebook", "exec"), namespace)
    original = namespace[name]().eval()
    extracted = {"CNN": CNN, "SNN": SNN}[name]().eval()
    extracted.load_state_dict(original.state_dict(), strict=True)
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(extracted(images), original(images), rtol=0, atol=0)
