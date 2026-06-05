"""Sanity-check the torch_env venv. Prints versions and runs a tiny GPU op if available."""

import platform
import sys


def main() -> int:
    print(f"python   = {sys.version.split()[0]}  ({platform.platform()})")

    try:
        import torch
        import torchvision
    except ImportError as e:
        print(f"FAIL: cannot import torch/torchvision -> {e}")
        return 1

    print(f"torch    = {torch.__version__}")
    print(f"torchvision = {torchvision.__version__}")
    print(f"cuda.is_available = {torch.cuda.is_available()}")
    print(f"torch.version.cuda = {torch.version.cuda}")
    print(f"torch.backends.cudnn.version = {torch.backends.cudnn.version()}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"gpu count = {n}")
        for i in range(n):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        print(f"matmul sanity sum = {y:.3f} (any finite number is fine)")
    else:
        print("NOTE: cuda not available — expected on login node, must run via srun for GPU test.")

    for pkg in ("timm", "einops", "decord"):
        try:
            mod = __import__(pkg)
            print(f"{pkg:10s} = {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"{pkg:10s} = MISSING")

    # kaggle: don't import (its module-level code calls sys.exit on missing creds);
    # just check that the package is installed.
    import importlib.util as _il
    spec = _il.find_spec("kaggle")
    print(f"{'kaggle':10s} = {'installed' if spec is not None else 'MISSING'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
