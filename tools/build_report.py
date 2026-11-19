"""Build the submission-format project report from saved, auditable results.

Requires reportlab and pypdf, plus licensed Times New Roman font files.
Use --font-dir to select the directory containing times.ttf and timesbd.ttf.
"""
import argparse
import json
from pathlib import Path
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from pypdf import PdfReader


def build(root, font_dir):
    root = Path(root)
    results = {name: json.loads((root/f"outputs/cifar10_pilot/{name}/summary.json").read_text()) for name in ("cnn", "snn")}
    tests = {name: json.loads((root/f"outputs/cifar10_pilot/{name}/test_metrics.json").read_text()) for name in results}
    c, s = results["cnn"], results["snn"]
    font_dir = Path(font_dir)
    pdfmetrics.registerFont(TTFont("TimesNewRoman", str(font_dir/"times.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRomanBold", str(font_dir/"timesbd.ttf")))
    pdfmetrics.registerFontFamily("TimesNewRoman", normal="TimesNewRoman", bold="TimesNewRomanBold", italic="TimesNewRoman", boldItalic="TimesNewRomanBold")
    body = ParagraphStyle("Body", fontName="TimesNewRoman", fontSize=12, leading=14.4,
                          alignment=TA_JUSTIFY, spaceAfter=7, splitLongWords=True)
    heading = ParagraphStyle("Heading", parent=body, fontName="TimesNewRomanBold", alignment=0,
                             spaceBefore=9, spaceAfter=6, keepWithNext=True)
    centered = ParagraphStyle("Centered", parent=body, alignment=TA_CENTER)
    sections = [
      ("Abstract", [
        "This project implements a compact comparison of conventional and spiking neural networks on CIFAR-10. The existing SNN uses leaky integrate-and-fire neurons, surrogate gradients and temporal effective batch normalization (TEBN). Our new contribution makes the comparison rerunnable through shared model code, seeded data partitions, validation-based checkpoint selection and saved evaluation artifacts. Fifteen behavioral tests pass. A new three-epoch CPU subset pilot verifies execution and artifact generation. It does not establish full-training performance. Accuracy and GPU-energy figures in the supplied presentation have no supporting raw files in the repository or its bundled development history and remain unverified."
      ]),
      ("I  Research Context", [
        "Sparse binary communication motivates SNNs as an alternative to conventional networks. Event-driven hardware can avoid some arithmetic when no spike arrives. A dense PyTorch simulation still executes tensor operations on zero inputs, so sparsity alone does not establish a GPU or system-energy advantage.",
        "Duan et al. introduce TEBN, which learns temporal scaling after normalization over samples from all time steps [1]. Their analysis connects this normalization to smoother optimization and gradient stabilization. The paper reports 92.65% for a seven-layer CNN at four time steps, and 94.70% for ResNet-19 at four time steps without cutout in Table 3. These are different architectures and settings from our compact implementation. Our study checks a simplified method implementation and comparison protocol, rather than reproducing those headline accuracies."
      ]),
      ("II  Model and Training Method", [
        "Both models contain six 3 by 3 convolutions with channel pairs 64, 128 and 256, with a 2 by 2 max pool after each pair. The classifier maps 4,096 features to 256 units and then ten logits. The CNN uses batch normalization and ReLU. It contains 2,197,706 trainable parameters. The SNN substitutes TEBN and LIF activations, with 2,197,730 parameters at T = 4. Its 24 additional scalars are six temporal-normalization layers times four steps.",
        "For the SNN, the same normalized image enters each time step. The membrane update is v<sub>t</sub> = 0.5u<sub>t-1</sub> + I<sub>t</sub>. A spike s<sub>t</sub> = H(v<sub>t</sub> - 1) resets the state through u<sub>t</sub> = v<sub>t</sub>(1 - s<sub>t</sub>). State starts at zero for every forward call. Backpropagation uses the triangular surrogate max(0, 1 - |v<sub>t</sub> - 1|). The implementation retains the original differentiable reset path.",
        "TEBN computes y<sub>t</sub> = p<sub>t</sub> BN(x<sub>t</sub>), pooling batch, time and spatial positions for each channel. Its learned p<sub>t</sub> multiplies both the normalized scale and offset [1]. The final classifier averages logits over time before cross-entropy loss. The optional plain-normalization baseline shares batch/time statistics without p<sub>t</sub>; it is not BNTT.",
        "The full-run defaults retain SGD with Nesterov momentum 0.9, initial learning rate 0.05, weight decay 0.0005 and a cosine schedule over 30 epochs. The pilot uses three epochs and batch size 16. These compute limits and the small pilot subset prevent a direct accuracy comparison with the paper or the original presentation."
      ]),
      ("III  Implementation Contributions", [
        "The baseline commit and its bundled ten-commit history attribute the original data, CNN, SNN, TEBN and exploratory measurement code to Ethan Vu. Marcus Vu's contribution adds the reproducible data and CNN experiment workflow, shared model extraction, evaluation and verification, executed pilot, and updated report and slides. Existing code and authorship remain traceable.",
        "The new data pipeline makes a seeded, stratified validation holdout exclusively from the 50,000 official training images [2]. Defaults produce 45,000 training and 5,000 validation examples. A separate dataset instance ensures validation receives no random crop, flip, color jitter or rotation. The official 10,000-image test partition remains separate. Original indices and a split digest are saved for each model.",
        "The original notebook selected the best checkpoint by test accuracy after every epoch. The new runner selects only by validation accuracy, keeps the earlier checkpoint on ties, and evaluates test labels after selection. Initializing the best score below every attainable accuracy ensures a checkpoint exists even when validation accuracy is zero. The final partial batch contributes its actual sample count to loss and accuracy.",
        "A run manifest records configuration, software versions, device, source-file digests and code revision. Each model saves epoch history, a checkpoint, held-out metrics, a confusion matrix, per-class precision/recall/F1, operation counts and temporal scales. A nonempty output directory is rejected to protect earlier evidence. Synthetic smoke runs carry an explicit synthetic label. Deterministic seeds cover Python, NumPy, PyTorch, the data sampler and workers, while cross-device/version equivalence remains an experimental limitation [3]."
      ]),
      ("IV  Measurement and Verification", [
        "CUDA synchronization bounds throughput and latency timing. Throughput includes the loader and host-to-device transfer; latency uses a warmed, preloaded single image. Training wall time includes validation and checkpoint I/O. NVML power samples map to the CUDA device UUID and integrate over timestamps. Missing readings remain null with an explanation. The CPU-only verification host cannot validate the CUDA/NVML path or produce measured GPU energy.",
        "Arithmetic energy is a separate model: E = N<sub>MAC</sub>c<sub>MAC</sub> + N<sub>AC</sub>c<sub>AC</sub>. We retain the original explicit cost assumptions of 4.6 pJ per MAC and 0.9 pJ per addition. Profiling counts actual time-expanded rows, convolution groups and partial batches. The first image layer remains dense. Later binary SNN inputs use a layer-specific activity-weighted addition estimate. Binary-valued ANN tensors do not automatically become spikes.",
        "The estimate excludes normalization, pooling, bias, neuron-state updates, memory movement and routing. Average input activity approximates convolutional fan-out and boundary effects. Therefore, the arithmetic result cannot predict end-to-end consumption on Loihi or another named device. The new report removes the supplied deck's unsupported chip-deployment comparisons.",
        "Fifteen tests cover stratified disjoint partitions, augmentation isolation, weighted metrics, zero predicted classes, LIF reset and surrogate gradients, TEBN statistics and gradients, both model backward passes, exact parameter counts, grouped/time-expanded operations, hook cleanup on failure, irregular power integration, missing CPU power, seeding and checkpoint/test routing. The integration test verifies that zero validation accuracy still produces a usable checkpoint."
      ]),
      ("V  Experiments and Results", [
        "The executed pilot uses the official CIFAR-10 dataset with 100 training, 50 validation and 50 test images per class: 1,000/500/500 in total. Both models use the same stored indices, seed 42, T = 4 for the SNN and three training epochs. Execution uses Windows, Python 3.13, PyTorch 2.11.0+cpu, torchvision 0.26.0+cpu and four CPU threads. Full configurations and hashes are stored in outputs/cifar10_pilot/run.json.",
        f"The selected CNN epoch is {c['best_epoch']} with {c['best_validation_accuracy_percent']:.2f}% validation accuracy. The SNN selects epoch {s['best_epoch']} at {s['best_validation_accuracy_percent']:.2f}%. Table I reports the held-out observations. These short-run accuracies describe this subset pilot only. They cannot support a claim that either architecture is superior after adequate training."
      ]),
      ("VI  Critical Analysis", [
        "The supplied deck reports 91.78% CNN and 89.98% SNN best test accuracy, plus GPU energy of 9.52 and 40.01 mJ per image. Both current notebooks have no saved outputs, and neither the tracked repository nor the bundled history contains corresponding CSV logs or checkpoints. These values are retained only as previously reported, unverified claims. Recreating their plots would falsely imply access to raw measurements. The new pilot is the only newly verified training evidence.",
        "Repeated test-based selection in the legacy workflow can produce optimistic best-epoch accuracy. Our validation holdout changes both the selection protocol and the available training size, so its test accuracy must not be pooled with that legacy metric. The pilot also uses fewer epochs, a smaller batch and a very small training subset. Low accuracy is evidence that the run is incomplete as a performance reproduction, not a failure of the artifact-generation checks.",
        "Only one seed and one CPU environment are measured. Wilson intervals describe finite test-sample uncertainty conditional on the selected model, not variation across training seeds. Timing depends on this host, batch size and data overhead. Firing rates of undertrained models cannot establish the sparsity of a converged SNN. An arithmetic estimate and measured GPU board energy cover different components and units. No neuromorphic device was deployed or measured."
      ]),
      ("VII  Remaining Experiments", [
        "Before the final presentation, run the full validation-based comparison for multiple seeds on the intended GPU and retain run manifests, histories, selected checkpoints and raw power records. Compare TEBN with the shared-statistics baseline under identical data and training settings. Any time-step study must train matching TEBN scale dimensions rather than truncating a fixed-T checkpoint. Report mean and variability, inspect confusion matrices, and distinguish convergence differences from measurement effects.",
        "The completed contribution provides a tested experiment and evidence workflow. Full-scale accuracy reproduction, GPU-energy verification, a measured normalization ablation and hardware deployment remain unfinished. The report and updated presentation make those boundaries explicit so the final project can be reviewed and rerun from the repository [4]."
      ]),
      ("References", [
        '[1] C. Duan et al., "Temporal Effective Batch Normalization in Spiking Neural Networks," NeurIPS, vol. 35, 2022. <link href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html" color="black">Publisher paper and supplement</link>.',
        '[2] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," University of Toronto, 2009. <link href="https://cave.cs.toronto.edu/kriz/cifar.html" color="black">Official CIFAR dataset description</link>.',
        '[3] PyTorch contributors, "Reproducibility," PyTorch documentation, accessed Nov. 18, 2026. <link href="https://docs.pytorch.org/docs/stable/notes/randomness.html" color="black">Randomness and deterministic algorithms</link>.',
        '[4] E. Vu and M. Vu, "Neuromorphic SNN Experiment," project repository, 2026. <link href="https://github.com/LittleSA1GON/Neuromorphic-SNN-Experiment" color="black">Source code and run artifacts</link>.'
      ])]
    output = root/"docs/project_report.pdf"
    output.parent.mkdir(exist_ok=True)
    doc = BaseDocTemplate(str(output), pagesize=(612, 792), leftMargin=36, rightMargin=36,
                          topMargin=36, bottomMargin=42, title="Reproducible SNN and ANN Comparison on CIFAR-10",
                          author="Ethan Vu and Marcus Vu")
    def page(canvas, document):
        canvas.saveState()
        canvas.setFont("TimesNewRoman", 12)
        canvas.drawCentredString(306, 22, str(document.page))
        if document.page == 1:
            canvas.setFont("TimesNewRomanBold", 12)
            canvas.drawCentredString(306, 755, "Reproducible SNN and ANN Comparison on CIFAR-10")
            canvas.setFont("TimesNewRoman", 12)
            canvas.drawCentredString(306, 735, "Ethan Vu and Marcus Vu")
            canvas.drawCentredString(306, 715, "CMPE 257   Project report and contribution update   November 18 2026")
        canvas.restoreState()
    def frames(top):
        return [Frame(x,42,261,top-42,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for x in (36,315)]
    doc.addPageTemplates([PageTemplate(id="first",frames=frames(689),onPage=page,autoNextPageTemplate="rest"),
                          PageTemplate(id="rest",frames=frames(756),onPage=page)])
    story=[]
    for title, paragraphs in sections:
        story.append(Paragraph(title, heading))
        story.extend(Paragraph(text,body) for text in paragraphs)
        if title == "V  Experiments and Results":
            data=[["Metric", "CNN", "SNN"],
                  ["Test accuracy %",f"{c['test_accuracy_percent']:.2f}",f"{s['test_accuracy_percent']:.2f}"],
                  ["Macro F1",f"{c['test_macro_f1']:.3f}",f"{s['test_macro_f1']:.3f}"],
                  ["Training seconds",f"{c['training_wall_seconds']:.1f}",f"{s['training_wall_seconds']:.1f}"],
                  ["Images per second",f"{c['images_per_second']:.1f}",f"{s['images_per_second']:.1f}"],
                  ["Latency ms",f"{c['single_image_latency_ms']:.2f}",f"{s['single_image_latency_ms']:.2f}"],
                  ["Arithmetic uJ/image",f"{c['arithmetic_energy_uj_per_image']:.1f}",f"{s['arithmetic_energy_uj_per_image']:.1f}"]]
            table=Table([[Paragraph(str(value),body) for value in row] for row in data],colWidths=[139,61,61],hAlign="LEFT")
            table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),2),
                                       ("RIGHTPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),4),
                                       ("BOTTOMPADDING",(0,0),(-1,-1),2),("LINEBELOW",(0,0),(-1,0),.6,"black")]))
            story.append(KeepTogether([Paragraph("Table I  Observed CPU subset pilot",heading),table]))
            intervals=[]
            for name in ("cnn","snn"):
                low,high=tests[name]["accuracy_wilson95_percent"]
                intervals.append(f"{name.upper()} {low:.2f}% to {high:.2f}%")
            story.append(Paragraph("Accuracy Wilson 95% intervals: "+"; ".join(intervals)+". GPU board energy is unavailable on this CPU host. Arithmetic values are estimates under the stated cost assumptions.",body))
    doc.build(story)
    reader=PdfReader(output)
    if len(reader.pages)>6:
        raise ValueError(f"report exceeds six pages: {len(reader.pages)}")
    (root/"docs/report_source.json").write_text(json.dumps({"date":"2026-11-18","format":"two columns; Times New Roman; 12 pt","sections":sections},indent=2)+"\n",encoding="utf-8")
    print(f"REPORT {output} PAGES {len(reader.pages)}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=".")
    parser.add_argument("--font-dir",default="C:/Windows/Fonts")
    args=parser.parse_args()
    build(args.root,args.font_dir)
