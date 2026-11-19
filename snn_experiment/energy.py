"""Activity-weighted arithmetic estimates separate from measured GPU energy."""
import torch
from torch import nn
from .models import LIFNeuron

MAC_JOULES = 4.6e-12
AC_JOULES = 0.9e-12


@torch.no_grad()
def profile_operations(model, loader, device, *, spiking=False, max_batches=5):
    """Count Conv/Linear work per original image, including observed time steps.

    Spike ACs use mean presynaptic activity, an approximation for convolutions
    with nonuniform activity and padding. Costs exclude normalization, pooling,
    neuron updates, memory movement, routing, bias and control. These are 45 nm
    arithmetic assumptions, not a named chip's end-to-end energy prediction.
    """
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    entries, firing, handles = {}, {}, []
    first_layer = next((name for name, layer in model.named_modules() if isinstance(layer, (nn.Conv2d, nn.Linear))), None)

    def operation_hook(name):
        def hook(layer, inputs, output):
            x = inputs[0]
            per_row = output[0].numel() * (layer.in_channels//layer.groups) * layer.kernel_size[0]*layer.kernel_size[1] if isinstance(layer, nn.Conv2d) else layer.in_features*layer.out_features
            dense = per_row * output.shape[0]
            entry = entries.setdefault(name, {"dense": 0, "active": 0.0, "nonzero": 0, "elements": 0,
                                               "binary": True, "spike_candidate": spiking and name != first_layer})
            nonzero = torch.count_nonzero(x).item()
            entry["dense"] += dense
            entry["active"] += dense * nonzero/x.numel()
            entry["nonzero"] += nonzero
            entry["elements"] += x.numel()
            entry["binary"] = entry["binary"] and bool(((x == 0) | (x == 1)).all())
        return hook

    def firing_hook(name):
        def hook(_layer, _inputs, output):
            entry = firing.setdefault(name, [0, 0])
            entry[0] += torch.count_nonzero(output).item()
            entry[1] += output.numel()
        return hook

    previous_mode = model.training
    model.eval()
    seen = 0
    try:
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                handles.append(layer.register_forward_hook(operation_hook(name)))
            elif isinstance(layer, LIFNeuron):
                handles.append(layer.register_forward_hook(firing_hook(name)))
        for index, (images, _) in enumerate(loader):
            model(images.to(device))
            seen += len(images)
            if index + 1 >= max_batches:
                break
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_mode)
    if seen == 0:
        raise ValueError("cannot profile an empty loader")
    layers = []
    for name, entry in entries.items():
        spike_input = entry["spike_candidate"] and entry["binary"]
        layers.append({"layer": name, "dense_macs_per_image": entry["dense"]/seen,
                       "macs_per_image": 0 if spike_input else entry["dense"]/seen,
                       "estimated_acs_per_image": entry["active"]/seen if spike_input else 0,
                       "input_activity": entry["nonzero"]/entry["elements"], "spike_input": spike_input})
    macs = sum(layer["macs_per_image"] for layer in layers)
    acs = sum(layer["estimated_acs_per_image"] for layer in layers)
    return {"profiled_images": seen, "macs_per_image": macs, "estimated_acs_per_image": acs,
            "arithmetic_energy_uj_per_image": (macs*MAC_JOULES + acs*AC_JOULES)*1e6,
            "mac_joules": MAC_JOULES, "ac_joules": AC_JOULES,
            "scope": "45nm arithmetic estimate; excludes memory, routing, normalization and neuron state",
            "layers": layers, "firing_rates": {name: count/total for name, (count, total) in firing.items()},
            "element_weighted_firing_rate": sum(pair[0] for pair in firing.values())/sum(pair[1] for pair in firing.values()) if firing else None}
