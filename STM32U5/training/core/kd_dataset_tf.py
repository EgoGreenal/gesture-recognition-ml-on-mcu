"""Day 10: thin re-export -- build_kd_dataset now lives in data_loader_tf.

The Day 9 implementation here did PIL decode in a Python generator and was
GIL-bound (~9 samples/sec aggregate even at num_shards=8). The Day 10 rewrite
moves decode + crop + resize into the tf.data graph via tf.io.decode_jpeg
(C++, GIL-free). This module is now just a shim so existing callers keep
working without code changes.
"""

from data_loader_tf import build_kd_dataset  # noqa: F401

__all__ = ["build_kd_dataset"]
