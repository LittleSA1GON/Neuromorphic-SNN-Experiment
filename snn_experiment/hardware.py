"""Synchronized timing and optional GPU board power with explicit missing data."""
import math
import statistics
import threading
import time
import torch


def integrate_power(samples, start, end):
    """Trapezoidal joules with constant extension at the window edges."""
    if len(samples) < 2 or end <= start:
        return None
    values = sorted((t, p) for t, p in samples if start <= t <= end)
    if len(values) < 2 or any(not math.isfinite(p) or p < 0 for _, p in values):
        return None
    values = [(start, values[0][1]), *values, (end, values[-1][1])]
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(values, values[1:]))


class DeviceMeter:
    """NVML board power includes overhead within the measurement window.

    A CUDA UUID must map to NVML. Missing/failed readings stay null, never zero.
    This does not measure CPU power, system energy or neuromorphic hardware.
    """
    def __init__(self, device, interval=0.1):
        self.device, self.interval = torch.device(device), interval
        self.samples, self.utilization = [], []
        self.unavailable_reason = "CPU execution has no GPU board-power measurement"
        self.nvml = None

    def _sample(self):
        try:
            power = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            self.samples.append((time.perf_counter(), power))
            self.utilization.append(utilization)
        except Exception as error:
            self.unavailable_reason = f"NVML sampling failed: {type(error).__name__}"
            self.failed = True

    def _poll(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.failed = False
        self.thread = None
        self.stop = threading.Event()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                uuid = str(torch.cuda.get_device_properties(self.device).uuid)
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                self.unavailable_reason = None
            except Exception as error:
                self.unavailable_reason = f"GPU UUID/power unavailable: {type(error).__name__}"
                if self.nvml is not None:
                    self.nvml.nvmlShutdown()
                self.nvml = None
        self.start = time.perf_counter()
        if self.nvml is not None:
            self._sample()
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            self._sample()
        end = time.perf_counter()
        seconds = end - self.start
        joules = None if self.failed else integrate_power(self.samples, self.start, end)
        self.result = {"wall_seconds": seconds, "energy_joules": joules,
                       "average_board_power_watts": joules/seconds if joules is not None else None,
                       "average_gpu_utilization_percent": statistics.mean(self.utilization) if self.utilization else None,
                       "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(self.device)/1e6 if self.device.type == "cuda" else None,
                       "power_samples": len(self.samples), "power_unavailable_reason": self.unavailable_reason}
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
        if self.device.type == "cuda" and joules is None and self.unavailable_reason is None:
            self.result["power_unavailable_reason"] = "insufficient valid NVML samples"


@torch.no_grad()
def benchmark(model, loader, device, repeats=30):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    previous_mode = model.training
    model.eval()
    try:
        images, _ = next(iter(loader))
        images = images.to(device)
        for _ in range(3):
            model(images)
        with DeviceMeter(device) as meter:
            count = 0
            for batch, _ in loader:
                model(batch.to(device))
                count += len(batch)
        one_image = images[:1]
        for _ in range(3):
            model(one_image)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(one_image)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter()-start)*1000/repeats
        result = dict(meter.result)
        result.update({"images_per_second": count/result["wall_seconds"],
                       "single_image_latency_ms": latency, "latency_repeats": repeats,
                       "measurement_scope": "throughput includes loader and host-to-device transfer",
                       "energy_mj_per_image": result["energy_joules"]*1000/count if result["energy_joules"] is not None else None})
        return result
    finally:
        model.train(previous_mode)
