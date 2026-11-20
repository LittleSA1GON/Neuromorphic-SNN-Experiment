"""Device selection checks without training or downloading CIFAR-10."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

notebook = json.loads((Path(__file__).resolve().parents[1] /
                       "notebooks/snn_vs_ann_tebn.ipynb").read_text(encoding="utf-8"))
source = next("".join(c["source"]) for c in notebook["cells"]
              if "def select_device(" in "".join(c["source"]))
function = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "select_device")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_setup", "exec"), namespace)
select_device = namespace["select_device"]


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(str(select_device()), "cpu")

    def test_explicit_cpu_does_not_initialize_cuda(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError):
            self.assertEqual(str(select_device("cpu")), "cpu")

    def test_explicit_cuda_never_silently_falls_back(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                select_device("cuda:0")

    def test_unsupported_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SNN_DEVICE"):
            select_device("mps")

    def test_selected_gpu_is_used_for_probe_and_synchronization(self):
        probe = torch.ones(1)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(torch.cuda, "set_device") as set_device, \
             patch.object(torch.cuda, "synchronize") as synchronize, \
             patch.object(torch, "ones", return_value=probe) as ones:
            selected = select_device("cuda:1")
            self.assertEqual(str(selected), "cuda:1")
            set_device.assert_called_once_with(selected)
            ones.assert_called_once_with(1, device=selected)
            synchronize.assert_called_once_with(selected)

    def test_invisible_gpu_index_is_rejected(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "not visible"):
                select_device("cuda:1")

    def test_kernel_failure_has_actionable_message(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(torch.cuda, "set_device"), \
             patch.object(torch, "ones", side_effect=RuntimeError("no kernel image")):
            with self.assertRaisesRegex(RuntimeError, "architecture support"):
                select_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
