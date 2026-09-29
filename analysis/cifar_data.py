"""CIFAR-10 loader matching pcalm.data's conventions, as an external caller.

pcalm/data.py is IDX-only and its DATASET_INFO is a module constant; the protocol
forbids editing the reference package, so this module reads the CIFAR-10 binary
release directly and returns exactly the tuple pcalm.data.load_dataset returns:
(x_train, y_train, x_test, y_test), images flattened and normalised per channel,
labels one-hot. The subset selection reuses pcalm.data.balanced_indices with the same
seed convention (test offset +17), so a CIFAR run differs from an MNIST run in the
data and nothing else.

Per-channel mean and std are the standard CIFAR-10 values.
"""
from __future__ import annotations
import tarfile
from pathlib import Path

import numpy as np

from pcalm.data import balanced_indices, one_hot

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
_REC = 3073          # 1 label byte + 3 * 32 * 32 image bytes
_ARCHIVE = "cifar-10-binary.tar.gz"
_TRAIN = [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)]
_TEST = ["cifar-10-batches-bin/test_batch.bin"]


def _read_members(data_dir, names):
    """Read records straight out of the tarball; nothing is unpacked to disk."""
    path = Path(data_dir) / _ARCHIVE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch the CIFAR-10 binary release into {data_dir}/.")
    xs, ys = [], []
    with tarfile.open(path, "r:gz") as tf:
        for name in names:
            buf = np.frombuffer(tf.extractfile(name).read(), dtype=np.uint8)
            rec = buf.reshape(-1, _REC)
            ys.append(rec[:, 0].astype(np.int64))
            # bytes are 1024 R then 1024 G then 1024 B, row-major within a channel
            xs.append(rec[:, 1:].reshape(-1, 3, 32, 32))
    return np.concatenate(xs), np.concatenate(ys)


def _normalize(images_u8):
    x = images_u8.astype(np.float32) / 255.0
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return x.reshape(x.shape[0], -1).astype(np.float32)   # NCHW -> flat 3072


def load_cifar10(*, train_subset, test_subset, seed, data_dir="data", output_dim=10):
    xtr_u8, ytr = _read_members(data_dir, _TRAIN)
    xte_u8, yte = _read_members(data_dir, _TEST)
    tr = balanced_indices(ytr, train_subset, seed)
    te = balanced_indices(yte, test_subset, seed + 17)
    return (_normalize(xtr_u8[tr]), one_hot(ytr[tr], output_dim),
            _normalize(xte_u8[te]), one_hot(yte[te], output_dim))
