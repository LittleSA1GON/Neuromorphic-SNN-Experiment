# Contribution record

This update adds a substantial data, baseline-training and evaluation contribution alongside the existing SNN implementation. The scope is judged by implemented and verified work. No historical commits or authors were changed, and no empty commits were created.

## Existing repository evidence

- The original remote baseline is `26ec7d26b32e02913c83c093911737a29e5807bb`, authored by LittleSA1GON.
- The checked-in Git bundle retains ten earlier development commits by Ethan Vu, ending at `4679328e6b169b51d8fd10dab2b0f0b90ce2e687`.
- Those commits record the original proposal, CIFAR preprocessing, CNN, LIF/surrogate implementation, SNN/TEBN, training/measurement and exploratory comparison work. Git records commit authorship; it does not establish each person's unrecorded work.
- None of the historical notebooks has saved output cells, and no tracked experiment logs or checkpoints support the full-training figures in the supplied slide PDF. `evidence_audit.json` records every checked revision and the two supplied PDFs' hashes.

## New work in Marcus's contribution

| Work | Implementation and evidence |
|---|---|
| Reproducible CIFAR preprocessing | Stratified holdout from official training data, separate transforms, deterministic samplers/workers, saved indices and split hashes in `snn_experiment/data.py` |
| CNN and matched-model experiment workflow | Configurable CPU/GPU CLI, preserved model topology, optimizer/schedule, validation-based selection, zero-score checkpoint handling, fresh-output protection and saved run metadata |
| Evaluation | Sample-weighted loss/accuracy, held-out test routing, confusion matrices, per-class precision/recall/F1, macro F1 and Wilson accuracy intervals |
| Measurement fixes | Synchronized timing, explicit unavailable GPU power, timestamped integration, grouped/time-expanded operation counts, cleanup of hooks and weighted firing counts |
| Verification | 15 passing tests, including exact output parity against immutable original CNN/SNN definitions and end-to-end checkpoint/evaluation checks |
| Executed experiment | Three-epoch CIFAR-10 CPU subset pilot, 1,000/500/500 train/validation/test, seed 42, batch 16, matching model split hashes |
| Documentation | Canonical notebook using the shared package, setup/reproduction commands, evidence audit, three-page report and editable 22-slide presentation |

## Observed pilot results

CNN held-out accuracy: **10.00%**. SNN held-out accuracy: **27.40%**. Both selected epoch 3 by validation accuracy. The small training subset and short budget do not establish final-model performance. The CNN remains near chance. GPU board energy is null because this host runs CPU-only PyTorch. Arithmetic estimates use the stated assumptions and exclude system overhead.

## What still needs experimental work

Full training on the intended GPU across multiple seeds, retained raw GPU measurements, and the normalization ablation remain unfinished. The supplied presentation's 91.78%/89.98% best-test figures are explicitly unverified. Prior test-selected accuracies and new validation-selected test accuracies use different protocols.

## Rebuilding deliverables

The report builder reads the saved pilot results and requires ReportLab, pypdf and Times New Roman font files:

```bash
python tools/build_report.py --font-dir PATH_TO_TIMES_NEW_ROMAN
```

The editable presentation builder uses the configured Codex Artifact Tool runtime. Set `ARTIFACT_TOOL_MODULE` to its `artifact_tool.mjs`, `PRESENTATIONS_SKILL_DIR` to the presentation skill, `ARTIFACT_PYTHON` to bundled Python, and `RUNTIME_NODE_MODULES` to bundled node_modules. Run `tools/build_presentation.mjs` with bundled Node. It renders each slide, exports a candidate, validates charts/tables/package layout and performs a first-party import before writing the final PPTX. Its finalizer requires a new final path when regenerating an already delivered file.

`report_source.json` and `presentation_source.json` retain the report narrative and slide outline with citations. The PDF report follows the supplied six-page maximum, two-column, 12-point Times New Roman requirement.
