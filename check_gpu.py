"""Tiny existing-PyTorch check. A failed nvidia-smi is not a CUDA verdict."""
from __future__ import annotations

import json
import subprocess
import sys

from hashutil import dump_json
from paths import LOGS, TORCH_PY


def main():
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    script = r"""
import json, torch
x = torch.arange(8, dtype=torch.float32)
cpu = float(x.sum())
info = {
    "torch": torch.__version__,
    "cuda_compiled": torch.version.cuda,
    "torch_cuda_is_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "cpu_sum": cpu,
}
if torch.cuda.is_available():
    y = x.to("cuda")
    info["gpu_sum"] = float(y.sum().item())
    info["cpu_gpu_agree"] = abs(info["gpu_sum"] - cpu) < 1e-6
print(json.dumps(info))
"""
    proc = subprocess.run([str(TORCH_PY), "-c", script], capture_output=True, text=True)
    report = {
        "nvidia_smi_returncode": smi.returncode,
        "nvidia_smi_stdout": (smi.stdout or "").strip(),
        "nvidia_smi_stderr": (smi.stderr or "").strip(),
        "nvidia_smi_note": "a failed nvidia-smi is not proof that CUDA is unavailable to this PyTorch build",
        "torch_ok": proc.returncode == 0,
        "torch_stdout": proc.stdout.strip(),
        "torch_stderr": proc.stderr.strip()[-1500:],
        "claimed_gpu_operator": False,
        "note": "existing NumPy methods are not GPU versions because a tensor wrapper was not written",
    }
    if proc.returncode == 0:
        report["torch"] = json.loads(proc.stdout)
    LOGS.mkdir(parents=True, exist_ok=True)
    dump_json(LOGS / "gpu_check.json", report)
    print(json.dumps(report, indent=2)[:2000])


if __name__ == "__main__":
    main()
