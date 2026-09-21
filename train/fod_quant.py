"""
fod_quant.py
============
Float features (from fod_features.py) -> standardised -> INT8 -> 16 packed 32-bit AXI words.

    z  = clip((x - mean) / std, -CLIP, +CLIP)          mean/std from the TRAINING set only
    q  = round(z * 127 / CLIP)   in [-127, 127]         int8
    word[i] = q[4i] | q[4i+1] << 8 | q[4i+2] << 16 | q[4i+3] << 24     (little-endian bytes)

Train your network on   z_q = q * CLIP / 127   (i.e. the de-quantised values), or feed q/127
directly, so the network sees exactly what the hardware will see.

Register map (from your block diagram - change if your IP differs):
    0x00        control (start)
    0x10        predicted class (read)
    0x40..0x7F  16 feature words
"""

import numpy as np

CLIP = 4.0                 # +-4 sigma -> +-127
N_FEATURES = 64
N_WORDS = N_FEATURES // 4

REG_CONTROL = 0x00
REG_RESULT = 0x10
REG_FEATURES = 0x40


class FeatureScaler:
    """Standardise + quantise. Fit on the training split, save with the model."""

    def __init__(self, mean=None, std=None, clip=CLIP):
        self.mean = mean
        self.std = std
        self.clip = clip

    # ---------- fitting / IO ----------
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0)
        # constant features (e.g. always 0) -> std=1 so they map to 0, not to NaN/inf
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return self

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, clip=np.float32(self.clip))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["std"], float(d["clip"]))

    # ---------- transforms ----------
    def standardize(self, X):
        """float features -> clipped z-scores."""
        z = (np.asarray(X, dtype=np.float32) - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip)

    def quantize(self, X):
        """float features -> int8 in [-127, 127]."""
        z = self.standardize(X)
        return np.rint(z * (127.0 / self.clip)).astype(np.int8)

    def dequantize(self, q):
        """int8 -> the z-scores the hardware effectively sees (use these for training)."""
        return np.asarray(q, dtype=np.float32) * (self.clip / 127.0)


# ============================================================
# PACKING FOR AXI-LITE
# ============================================================
def pack_words(q):
    """int8 vector of 64 -> uint32 array of 16 (feature k lives in byte k%4 of word k//4)."""
    q = np.asarray(q, dtype=np.int8).reshape(N_WORDS, 4)
    u = q.view(np.uint8).astype(np.uint32)
    return u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16) | (u[:, 3] << 24)


def unpack_words(words):
    """Inverse of pack_words (for testing)."""
    w = np.asarray(words, dtype=np.uint32)
    b = np.stack([(w >> s) & 0xFF for s in (0, 8, 16, 24)], axis=1).astype(np.uint8)
    return b.reshape(-1).view(np.int8)


def classify_on_fpga(ip, q, poll=True):
    """
    Write 16 words, start the accelerator, return the predicted class.

    ip : pynq.Overlay IP handle exposing .write(offset, value) / .read(offset)
         e.g.  ip = overlay.cnn_accel   (name depends on your block design)
    """
    words = pack_words(q)
    for i, w in enumerate(words):
        ip.write(REG_FEATURES + 4 * i, int(w))
    ip.write(REG_CONTROL, 1)                       # start pulse (adapt to your IP's protocol)
    if poll:
        # optional: wait for a done bit if your IP has one (e.g. bit 1 of REG_CONTROL)
        pass
    return int(ip.read(REG_RESULT))


if __name__ == "__main__":
    # round-trip self-test
    rng = np.random.default_rng(0)
    q = rng.integers(-127, 128, 64).astype(np.int8)
    assert np.array_equal(unpack_words(pack_words(q)), q)
    print("pack/unpack round trip OK; first words:", [hex(int(w)) for w in pack_words(q)[:3]])
