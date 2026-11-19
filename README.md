# Neuromorphic SNN Experiment

CMPE 257 project by Ethan Vu and Marcus Vu. We compare a matched CNN and directly trained spiking network on CIFAR-10, using the temporal normalization idea from [Duan et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html).

The current contribution turns the original notebook into a reproducible experiment package. Model topology and neuron dynamics are preserved. Checkpoints now use validation accuracy, with test evaluation after selection. The original exploratory notebook remains available as `SNN_vs_ANN_TEBN_Proposal.ipynb`.

## Evidence status

The supplied presentation reports **91.78% CNN** and **89.98% SNN** best test accuracy. The tracked repository and its bundled ten-commit history contain no raw results, checkpoints or executed notebook outputs behind those claims. They are **previously reported and unverified**, not reproduced results. See [the evidence audit](docs/evidence_audit.json).

`outputs/cifar10_pilot/` contains a new **three-epoch CPU subset pilot** with 1,000 training, 500 validation and 500 test images. These files demonstrate the new protocol and contain actual observations. They do not establish full-training accuracy, a TEBN advantage or neuromorphic hardware efficiency. Synthetic runs are separately marked `smoke_only`.

## Setup

Use Python 3.11 or later. Create a virtual environment and install a compatible PyTorch/torchvision pair from [PyTorch's installation instructions](https://pytorch.org/get-started/locally/) for your hardware, then:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The committed verification used Python 3.13, PyTorch 2.11.0+cpu and torchvision 0.26.0+cpu on Windows. `requirements-tested.txt` records the tested analysis/test versions; it is not a universal CUDA installation recipe. GPU compatibility depends on both GPU architecture and the particular PyTorch build. The package uses one visible device.

## Run

Run commands from the repository root. Every output directory must be new or empty. Dataset downloading is opt-in.

```bash
# Installation check without network or CIFAR-10 (one synthetic epoch).
python -m snn_experiment.run --smoke --output outputs/smoke_local

# The exact committed CPU subset-pilot configuration.
python -m snn_experiment.run --model all --epochs 3 --batch-size 16 --train-per-class 100 --validation-per-class 50 --test-per-class 50 --threads 4 --benchmark-repeats 10 --download --output outputs/pilot_repeat

# Full comparison: 45k train / 5k validation / 10k official test.
python -m snn_experiment.run --epochs 30 --batch-size 128 --download --output outputs/full_seed42

# Optional shared-statistics BN ablation, same data split and seed.
python -m snn_experiment.run --model snn_plain --epochs 30 --download --output outputs/plain_seed42
```

`--model cnn` or `--model snn` runs one model. `--seed`, `--timesteps`, `--workers`, `--device` and other options are listed by `--help`. Reduce batch size if memory is insufficient. The canonical notebook, `notebooks/snn_vs_ann_tebn.ipynb`, calls the same package. On Kaggle, clone/attach the full repository before running it.

## Protocol

- A seeded, stratified split takes validation data exclusively from the official training partition. Validation and test use tensor conversion and normalization only. Training retains crop, flip, color jitter and rotation.
- Both models use SGD with Nesterov momentum 0.9, initial learning rate 0.05, weight decay 0.0005 and a cosine schedule. The full default uses 30 epochs. The compact six-convolution topology is a simplified reproduction, not the paper's ResNet-19.
- The CNN has 2,197,706 parameters. TEBN adds 24 time scales at T=4, giving the SNN 2,197,730 parameters. Spikes use a triangular surrogate, leak 0.5 and threshold 1 with hard reset. Classifier logits are averaged over time.
- The first checkpoint is saved even at zero validation accuracy. Ties retain the earlier epoch. Test labels never select the checkpoint.
- Fixed seeds cover Python, NumPy, PyTorch, sampling and data workers. Deterministic algorithms are requested. Identical results across software releases and devices are not guaranteed, so environment and source hashes are recorded.

## Saved outputs

Each run has `run.json` and `comparison.csv`. Each model directory has:

| File | Contents |
|---|---|
| `split.json` | Original dataset indices, sizes, normalization and split hash |
| `history.csv` | Training and validation metrics for every epoch |
| `best.pt` | Selected state, epoch, model configuration and split hash (ignored by Git) |
| `test_metrics.json` | Held-out accuracy, Wilson interval, macro F1 and confusion matrix |
| `per_class.csv` | Precision, recall, F1 and support for every class |
| `inference.json` | Warmed throughput and single-image latency with measurement scope |
| `training_hardware.json` | Training-window time and optional GPU board measurements |
| `operations.json` | Per-layer arithmetic estimates and sample-weighted spike activity |
| `temporal_scales.json` | Learned p[t] for every TEBN layer |
| `summary.json` | Comparison metrics with dataset and protocol labels |

Training wall time includes validation and checkpoint I/O. Throughput includes loading and host-to-device transfer; single-image latency uses a preloaded tensor. CUDA is synchronized around timing. NVML board power is integrated over timestamps and tied to the CUDA device UUID. Missing measurements are null with a reason. The CUDA/NVML path has not been exercised on this CPU-only host.

Arithmetic energy uses **4.6 pJ/MAC** and **0.9 pJ/AC** as explicit legacy 45 nm assumptions. Counts include observed time steps, convolution groups and partial batches. The first image layer remains dense; subsequent binary SNN inputs use activity-weighted addition estimates. Nonbinary inputs remain MACs. Layer activity is an approximation for convolutional fan-out. The estimate excludes normalization, pooling, bias, neuron state, memory traffic and routing. It is separate from GPU board energy and does not predict Loihi or another named chip's system energy.

## Repository layout

```text
snn_experiment/                 models, data, training and evaluation package
notebooks/snn_vs_ann_tebn.ipynb  notebook interface to the same package
SNN_vs_ANN_TEBN_Proposal.ipynb    original exploratory implementation
tests/                          behavioral and integration checks
outputs/cifar10_pilot/           actual CPU subset-pilot evidence
docs/proposal.pdf                original proposal
docs/evidence_audit.json         provenance of supplied-slide claims
docs/project_report.pdf         two-column 12 pt Times New Roman report
docs/presentation_updated.pptx   editable updated presentation
tools/                          report and presentation builders
```

## Contributions

The existing baseline commit and bundled development history credit Ethan Vu with the original data/CNN/SNN/TEBN notebook, measurement utilities and comparison code. Marcus's new contribution covers the reproducible data and CNN experiment workflow, shared model extraction, held-out evaluation, corrected accounting, verification, executed pilot and report/presentation update. This extends the implementation without retroactively assigning earlier work to a different author. See [the contribution record](docs/contributions.md).
