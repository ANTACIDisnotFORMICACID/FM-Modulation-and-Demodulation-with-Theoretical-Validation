"""Raw IQ file adapters with explicit format and rate metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FORMATS = ("cf32le", "cs8", "ci16le")


class FileIQSource:
    """Read blocks from raw complex-float or interleaved integer IQ files."""

    def __init__(self, path, *, sample_rate=None, fmt=None, center_freq=None):
        self.path = Path(path)
        meta_path = self.path.with_suffix(self.path.suffix + ".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.sample_rate = int(sample_rate if sample_rate is not None else meta.get("sample_rate", 0))
        self.fmt = fmt if fmt is not None else meta.get("fmt")
        self.center_freq = center_freq if center_freq is not None else meta.get("center_freq")
        if self.sample_rate <= 0 or self.fmt not in FORMATS:
            raise ValueError("IQ source requires sample_rate and fmt (cf32le/cs8/ci16le)")
        self.file = self.path.open("rb")
        self.closed = False

    def read(self, num_samples):
        if self.closed or num_samples <= 0:
            raise ValueError("source closed or num_samples is not positive")
        if self.fmt == "cf32le":
            return np.fromfile(self.file, dtype="<c8", count=num_samples)
        dtype = "i1" if self.fmt == "cs8" else "<i2"
        raw = np.fromfile(self.file, dtype=dtype, count=2*num_samples)
        if len(raw) % 2:
            raise ValueError("truncated IQ pair at end of file")
        scale = 128.0 if self.fmt == "cs8" else 32768.0
        pairs = raw.reshape(-1, 2).astype(np.float32)/scale
        return pairs[:, 0] + 1j*pairs[:, 1]

    def close(self):
        if not self.closed:
            self.file.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FileIQSink:
    """Write little-endian complex64 IQ plus adjacent JSON metadata."""

    def __init__(self, path, *, sample_rate, center_freq=None, source="synthetic"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.center_freq = center_freq
        self.source = source
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.file = self.path.open("wb")
        self.samples_written = 0
        self.closed = False

    def write(self, samples):
        if self.closed:
            raise ValueError("sink is closed")
        z = np.asarray(samples)
        if z.ndim != 1 or not np.iscomplexobj(z) or not np.all(np.isfinite(z)):
            raise ValueError("write expects finite one-dimensional complex IQ")
        z.astype("<c8").tofile(self.file)
        self.samples_written += len(z)

    def close(self):
        if not self.closed:
            self.file.close()
            meta = {"sample_rate": self.sample_rate, "center_freq": self.center_freq,
                    "fmt": "cf32le", "sample_count": self.samples_written,
                    "source": self.source}
            self.path.with_suffix(self.path.suffix + ".json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SyntheticIQSource:
    """In-memory source for tests and future hardware adapter contract."""

    def __init__(self, samples, sample_rate):
        self.samples = np.asarray(samples, dtype=np.complex64)
        self.sample_rate = int(sample_rate)
        self.fmt = "cf32le"
        self.center_freq = None
        self.position = 0

    def read(self, num_samples):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        out = self.samples[self.position:self.position+num_samples]
        self.position += len(out)
        return out

    def close(self):
        pass


class MemoryIQSink:
    def __init__(self, sample_rate):
        self.sample_rate = int(sample_rate)
        self.fmt = "cf32le"
        self.blocks = []

    def write(self, samples):
        self.blocks.append(np.asarray(samples, dtype=np.complex64).copy())

    def close(self):
        pass

    def result(self):
        return np.concatenate(self.blocks) if self.blocks else np.empty(0, dtype=np.complex64)
