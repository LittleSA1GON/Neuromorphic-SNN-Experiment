# Neuromorphic SNN Experiment

CMPE 257 project by Ethan Vu and Marcus Vu.

ANNs need a lot of power to train and run, and that cost keeps growing as models get bigger. Spiking neural networks (SNNs) only do work when a neuron fires, which could make them a much cheaper alternative. In this project we train a CNN and an SNN with the same layers on CIFAR-10 and compare how they do on accuracy, speed, GPU usage and energy.

The SNN follows [*Temporal Effective Batch Normalization in Spiking Neural Networks*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html) (Duan et al., NeurIPS 2022):

- **LIF neurons**: leaky integrate-and-fire with a hard reset
- **Surrogate gradients**: a triangle-shaped gradient so the spiking step can be trained with backprop
- **TEBN**: batch norm statistics shared across all time steps, plus a learned scale for each time step

## What we measure

| Metric | How |
|---|---|
| Accuracy | Top-1 on the CIFAR-10 test set |
| Training time | Wall-clock time per epoch and in total |
| Inference speed | Images per second, and latency for a single image |
| GPU usage | Peak memory and average utilization |
| Power / energy | GPU power sampled with NVML during training and inference |
| Theoretical energy | Operation count (MACs vs. spike-driven additions) to estimate cost on neuromorphic chips |
| Spike activity | Firing rate of each SNN layer |

## Results (Kaggle, Tesla T4, 30 epochs, T = 4)

| Metric | CNN (ANN) | SNN (TEBN) | SNN / CNN |
|---|---|---|---|
| Best test accuracy | 91.78% | 89.98% | 0.98x |
| Training time | 11.6 min | 54.8 min | 4.73x |
| Training energy (GPU) | 42.4 kJ | 214.8 kJ | 5.06x |
| Peak GPU memory (train) | 488 MB | 3,360 MB | 6.89x |
| Inference throughput | 6,638 img/s | 1,631 img/s | 0.25x |
| Single-image latency | 1.26 ms | 5.85 ms | 4.64x |
| Measured GPU energy / image | 9.52 mJ | 40.01 mJ | 4.20x |
| Est. neuromorphic energy / image | 707.6 µJ | 81.9 µJ | 0.12x |
| Average firing rate | - | 8.53% | - |

The SNN comes within 1.8 points of the CNN. On a GPU it costs about 4x more, because every layer runs once per time step and the GPU can't skip neurons that don't spike. With only about 8.5% of neurons firing, the operation-count estimate says it would use about 8.6x less energy on neuromorphic hardware.

The best epoch was picked by test accuracy, so these accuracies are slightly optimistic. A held-out validation split would fix that.

## Repo layout

```
notebooks/snn_vs_ann_tebn.ipynb   the full experiment
outputs/                          results table and plots from our runs
docs/proposal.pdf                 project proposal
requirements.txt                  packages for running locally
```

## Running it

**Kaggle (what we use):**
1. Upload `notebooks/snn_vs_ann_tebn.ipynb` to Kaggle.
2. Settings: Accelerator **GPU T4** (the notebook uses one GPU), Internet **on**. Avoid the P100; newer PyTorch versions don't support it.
3. Run once with `QUICK_RUN = True` to check everything works, then set it to `False` and use *Save Version -> Save & Run All*.

**Locally (NVIDIA CUDA GPU or CPU):**
Create a virtual environment, then install a matching `torch` / `torchvision` pair using the [official PyTorch selector](https://pytorch.org/get-started/locally/) for your OS, GPU and driver. Install it into the same Python environment used by the notebook kernel. A CPU-only PyTorch build cannot run CUDA.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# Install the compatible PyTorch pair from the official selector first.
python -m pip install -r requirements.txt
$env:SNN_DEVICE = "cuda:0"
$env:SNN_BATCH_SIZE = "64"
$env:SNN_QUICK_RUN = "1"
jupyter notebook notebooks/snn_vs_ann_tebn.ipynb
```

The setup cell reports the device, PyTorch/CUDA build and GPU name, then executes a small CUDA operation before training. If it fails, check the selected Jupyter kernel, driver and wheel compatibility. Remove `SNN_QUICK_RUN` or set it to `0` for the full run. Lower `SNN_BATCH_SIZE` if the SNN exceeds GPU memory.

| Environment variable | Default and purpose |
|---|---|
| `SNN_DEVICE` | `auto` chooses CUDA if available, otherwise CPU; `cpu` forces CPU; `cuda:N` requires a visible GPU index |
| `SNN_BATCH_SIZE` | `128`; reduce for smaller local GPUs |
| `SNN_NUM_WORKERS` | `0` on Windows notebooks to avoid worker-spawn issues; `2` elsewhere |
| `SNN_QUICK_RUN` | `0`; `1` uses the existing small-subset, two-epoch check |

For a GTX 1080 Ti, [NVIDIA lists compute capability 6.1](https://developer.nvidia.com/cuda/gpus/legacy). Use a wheel that supports the card; a CUDA version label alone does not establish compatibility. The setup prints `torch.cuda.get_arch_list()` and checks an actual operation. If a current wheel omits the architecture, consult [PyTorch's previous-version builds](https://pytorch.org/get-started/previous-versions/). This notebook uses one selected device, including when multiple GPUs are available. GPU power readings are optional and enabled only when NVML can identify that same GPU by UUID.

Run the focused setup tests with `python -m unittest discover -s tests -p test_local_gpu.py -v`.

Marcus's speaker notes for slides 2, 3, 4 part 1, 5, 6 and 10 are in [PDF](docs/speaker_notes.pdf) and [editable text](docs/speaker_notes.txt).

CIFAR-10 downloads automatically the first time you run it. Results and plots are saved to `outputs/` (on Kaggle, download them from the Output tab and put them in `outputs/` here).

## Notes

- Training runs for 30 epochs instead of the paper's 300, so accuracy will be lower than the paper's results.
- Both models use the same optimizer and learning rate (SGD, lr 0.05, cosine schedule).
- Measured GPU power will probably favor the CNN, since a GPU doesn't skip work for neurons that don't spike. The theoretical energy estimate shows what the SNN would cost on neuromorphic hardware.

## Team

| Ethan | Marcus |
|---|---|
| SNN and TEBN implementation | CIFAR-10 preprocessing |
| Spike activity and SNN experiments | CNN baseline |
| Metrics and comparison | Metrics and comparison |
