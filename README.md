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
2. Settings: Accelerator **GPU T4 x2**, Internet **on**. Avoid the P100; newer PyTorch versions don't support it.
3. Run once with `QUICK_RUN = True` to check everything works, then set it to `False` and use *Save Version -> Save & Run All*.

**Locally (GTX 1080 Ti):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
jupyter notebook notebooks/snn_vs_ann_tebn.ipynb
```
The cu126 build is needed for Pascal cards like the 1080 Ti. If the SNN runs out of memory, lower `BATCH_SIZE` to 64.

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
