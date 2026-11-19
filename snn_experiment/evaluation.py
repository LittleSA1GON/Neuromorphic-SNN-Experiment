"""Sample-weighted metrics and confusion matrices without extra dependencies."""
import math
import torch
from torch import nn


def classification_report(confusion):
    confusion = torch.as_tensor(confusion, dtype=torch.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or (confusion < 0).any():
        raise ValueError("confusion must be a nonnegative square count matrix")
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    correct = confusion.diag()
    total = int(support.sum())
    if total == 0:
        raise ValueError("cannot evaluate an empty dataset")
    precision = correct.double() / predicted.clamp(min=1)
    recall = correct.double() / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-15)
    probability = float(correct.sum()) / total
    z = 1.959963984540054
    denominator = 1 + z*z/total
    center = (probability + z*z/(2*total)) / denominator
    half = z * math.sqrt(probability*(1-probability)/total + z*z/(4*total*total)) / denominator
    return {"samples": total, "correct": int(correct.sum()), "accuracy_percent": 100*probability,
            "accuracy_wilson95_percent": [100*(center-half), 100*(center+half)],
            "macro_f1": float(f1.mean()), "confusion_matrix": confusion.tolist(),
            "per_class": [{"class_index": i, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i in range(len(support))]}


@torch.no_grad()
def evaluate(model, loader, device):
    previous_mode = model.training
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    total_loss = 0.0
    try:
        for images, labels in loader:
            labels = labels.to(device)
            logits = model(images.to(device))
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            encoded = (labels * 10 + logits.argmax(1)).cpu()
            confusion += torch.bincount(encoded, minlength=100).reshape(10, 10)
    finally:
        model.train(previous_mode)
    result = classification_report(confusion)
    result["loss"] = total_loss / result["samples"]
    return result
