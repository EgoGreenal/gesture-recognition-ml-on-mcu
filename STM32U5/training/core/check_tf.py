"""Sanity-check the tf_env venv. Prints versions, lists GPU devices, runs a tiny op."""

import os
import platform
import sys

# Quiet TF startup noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> int:
    print(f"python   = {sys.version.split()[0]}  ({platform.platform()})")

    try:
        import tensorflow as tf
    except ImportError as e:
        print(f"FAIL: cannot import tensorflow -> {e}")
        return 1

    print(f"tensorflow = {tf.__version__}")
    try:
        import keras  # standalone keras shipped alongside TF
        print(f"keras     = {keras.__version__}")
    except Exception as e:
        print(f"keras     = (lazy via tf, not directly importable: {type(e).__name__})")
    print(f"is_built_with_cuda = {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"gpu count = {len(gpus)}")
    for g in gpus:
        print(f"  {g}")

    if gpus:
        with tf.device("/GPU:0"):
            x = tf.random.normal((1024, 1024))
            y = tf.reduce_sum(tf.linalg.matmul(x, x)).numpy()
            print(f"matmul sanity sum = {float(y):.3f} (any finite number is fine)")
    else:
        print("NOTE: no GPU visible — expected on login node, must run via srun for GPU test.")

    try:
        import tensorflow_model_optimization as tfmot
        print(f"tensorflow_model_optimization = {tfmot.__version__}")
    except ImportError:
        print("tensorflow_model_optimization = MISSING")

    for pkg in ("einops",):
        try:
            mod = __import__(pkg)
            print(f"{pkg:10s} = {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"{pkg:10s} = MISSING")

    import importlib.util as _il
    spec = _il.find_spec("kaggle")
    print(f"{'kaggle':10s} = {'installed' if spec is not None else 'MISSING'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
