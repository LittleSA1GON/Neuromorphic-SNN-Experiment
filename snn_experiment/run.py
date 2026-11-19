"""Run the matched CNN/SNN with auditable data and checkpoint selection."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import torch
from torch import nn
from .data import CLASS_NAMES, make_loaders
from .energy import profile_operations
from .evaluation import evaluate
from .hardware import DeviceMeter, benchmark
from .models import TEBN, make_model


@dataclass
class Config:
    model: str = "all"
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.05
    weight_decay: float = 5e-4
    timesteps: int = 4
    seed: int = 42
    workers: int = 0
    threads: int = 4
    validation_per_class: int = 500
    train_per_class: int | None = None
    test_per_class: int | None = None
    data_dir: str = "data"
    output: str = "outputs/experiment"
    device: str = "auto"
    download: bool = False
    smoke: bool = False
    benchmark_repeats: int = 30

    def validate(self):
        if self.model not in ("all", "cnn", "snn", "snn_plain"):
            raise ValueError("unknown model")
        if min(self.epochs, self.batch_size, self.timesteps, self.threads, self.validation_per_class, self.benchmark_repeats) < 1 or self.workers < 0:
            raise ValueError("epochs, batch size, time steps, threads and sizes must be positive; workers nonnegative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if any(value is not None and value < 1 for value in (self.train_per_class, self.test_per_class)):
            raise ValueError("subset sizes must be positive")
        if self.smoke and self.epochs != 1:
            raise ValueError("smoke runs use exactly one epoch")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def environment():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    import torchvision
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "source_sha256": source_hashes}


def run_model(config, name, device, directory):
    directory.mkdir()
    seed_everything(config.seed)
    loaders, split_manifest = make_loaders(config)
    write_json(directory/"split.json", split_manifest)
    model = make_model(name, config.timesteps).to(device)
    # Model initialization consumes different RNG sequences. Reset before
    # augmentation/training so the models start with the same data RNG.
    seed_everything(config.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,
                                nesterov=True, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history, best, best_epoch = [], -float("inf"), None
    with DeviceMeter(device) as meter:
        for epoch in range(1, config.epochs + 1):
            start = time.perf_counter()
            model.train()
            loss_sum, correct, count = 0.0, 0, 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for images, labels in loaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()*len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
            validation = evaluate(model, loaders["validation"], device)
            scheduler.step()
            history.append({"epoch": epoch, "learning_rate": learning_rate, "train_loss": loss_sum/count,
                            "train_accuracy_percent": 100*correct/count, "validation_loss": validation["loss"],
                            "validation_accuracy_percent": validation["accuracy_percent"],
                            "training_and_validation_seconds": time.perf_counter()-start})
            if validation["accuracy_percent"] > best:
                best, best_epoch = validation["accuracy_percent"], epoch
                temporary = directory/"best.tmp.pt"
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_accuracy_percent": best,
                            "config": asdict(config), "model_name": name, "split_sha256": split_manifest["split_sha256"]}, temporary)
                os.replace(temporary, directory/"best.pt")
            with (directory/"history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(f"{name} epoch {epoch}/{config.epochs} train={100*correct/count:.2f}% validation={validation['accuracy_percent']:.2f}%", flush=True)
    checkpoint = torch.load(directory/"best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    # Only validation chooses the checkpoint. Test labels are consumed here.
    testing = evaluate(model, loaders["test"], device)
    for item in testing["per_class"]:
        item["class_name"] = CLASS_NAMES[item["class_index"]]
    write_json(directory/"test_metrics.json", testing)
    with (directory/"per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(testing["per_class"][0]))
        writer.writeheader()
        writer.writerows(testing["per_class"])
    inference = benchmark(model, loaders["test"], device, repeats=config.benchmark_repeats)
    operations = profile_operations(model, loaders["test"], device, spiking=name != "cnn")
    write_json(directory/"inference.json", inference)
    write_json(directory/"operations.json", operations)
    scales = {name: layer.timestep_scale.detach().cpu().flatten().tolist() for name, layer in model.named_modules() if isinstance(layer, TEBN)}
    write_json(directory/"temporal_scales.json", scales)
    result = {"model": name, "dataset": split_manifest["dataset"], "scope": split_manifest.get("scope", "smoke_only"),
              "train_samples": split_manifest["counts"]["train"], "validation_samples": split_manifest["counts"]["validation"],
              "test_samples": testing["samples"], "epochs": config.epochs, "seed": config.seed,
              "parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
              "best_validation_accuracy_percent": best, "test_accuracy_percent": testing["accuracy_percent"],
              "test_macro_f1": testing["macro_f1"], "training_wall_seconds": meter.result["wall_seconds"],
              "images_per_second": inference["images_per_second"], "single_image_latency_ms": inference["single_image_latency_ms"],
              "gpu_energy_mj_per_image": inference["energy_mj_per_image"],
              "arithmetic_energy_uj_per_image": operations["arithmetic_energy_uj_per_image"],
              "split_sha256": split_manifest["split_sha256"], "device": str(device)}
    write_json(directory/"training_hardware.json", meter.result)
    write_json(directory/"summary.json", result)
    model.to("cpu")
    del model, checkpoint, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(config):
    config.validate()
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    directory = Path(config.output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "config": asdict(config), "environment": environment(), "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "selection_metric": "validation_accuracy_percent", "test_selection": False,
                "smoke_only": config.smoke}
    write_json(directory/"run.json", metadata)
    try:
        names = ("cnn", "snn") if config.model == "all" else (config.model,)
        results = [run_model(config, name, device, directory/name) for name in names]
        with (directory/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        metadata["status"] = "completed"
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory/"run.json", metadata)
        return results
    except Exception as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(directory/"run.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "cnn", "snn", "snn_plain"), default="all")
    for field in ("epochs", "batch_size", "timesteps", "seed", "workers", "threads", "validation_per_class", "benchmark_repeats"):
        parser.add_argument("--"+field.replace("_", "-"), type=int, default=getattr(Config, field))
    for field in ("train_per_class", "test_per_class"):
        parser.add_argument("--"+field.replace("_", "-"), type=int, default=None)
    for field in ("learning_rate", "weight_decay"):
        parser.add_argument("--"+field.replace("_", "-"), type=float, default=getattr(Config, field))
    for field in ("data_dir", "output", "device"):
        parser.add_argument("--"+field.replace("_", "-"), default=getattr(Config, field))
    parser.add_argument("--download", action="store_true", help="allow downloading official CIFAR-10")
    parser.add_argument("--smoke", action="store_true", help="one synthetic epoch; never a CIFAR-10 result")
    arguments = vars(parser.parse_args())
    if arguments["smoke"]:
        arguments.update(epochs=1, batch_size=4, benchmark_repeats=3)
    results = run(Config(**arguments))
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
