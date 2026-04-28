#!/usr/bin/env python3
"""Detect CUDA compute capability from available GPUs and set TORCH_CUDA_ARCH_LIST."""

import subprocess
import sys


def detect_compute_capabilities():
    """Detect GPU compute capabilities using nvidia-smi or PyTorch."""
    caps = set()

    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().split("\n"):
            if line:
                # Convert "12.0" to "12.0"
                caps.add(line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try PyTorch as fallback
    if not caps:
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cap = f"{props.major}.{props.minor}"
                caps.add(cap)
        except ImportError:
            pass

    return sorted(caps)


def main():
    caps = detect_compute_capabilities()

    if not caps:
        # Default to common architectures if no GPU detected
        caps = ["8.0", "8.6", "9.0"]
        print(f"# No GPU detected, using defaults: {';'.join(caps)}", file=sys.stderr)
    else:
        print(f"# Detected GPUs: {';'.join(caps)}", file=sys.stderr)

    # Output the TORCH_CUDA_ARCH_LIST value
    print(";".join(caps))


if __name__ == "__main__":
    main()
