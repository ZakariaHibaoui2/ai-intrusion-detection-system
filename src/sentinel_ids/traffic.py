"""Synthetic network-flow generator.

Produces labelled flows whose statistics follow the behaviour of each traffic class
(e.g. port scans touch many ports with tiny SYN-only flows, exfiltration pushes large
uploads to a single host). It lets the whole pipeline be trained and tested anywhere
without shipping a multi-GB capture. Swap in real NetFlow/CIC-IDS features with the same
column names to train on production data.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FEATURES, LABELS

DEFAULT_MIX = {
    "benign": 0.70,
    "dos": 0.08,
    "portscan": 0.07,
    "bruteforce": 0.06,
    "exfiltration": 0.04,
    "botnet": 0.05,
}


def _benign(rng: np.random.Generator, n: int) -> np.ndarray:
    duration = rng.lognormal(0.5, 1.2, n)
    packets = rng.lognormal(3.0, 1.0, n).astype(int) + 2
    src_bytes = packets * rng.normal(420, 150, n).clip(60)
    dst_bytes = src_bytes * rng.lognormal(1.2, 0.8, n)
    return np.column_stack([
        duration, src_bytes, dst_bytes, packets, packets / np.maximum(duration, 0.01),
        (src_bytes + dst_bytes) / packets, rng.uniform(0.01, 0.08, n), rng.uniform(0, 0.05, n),
        rng.integers(1, 4, n), rng.integers(1, 6, n), rng.choice([0, 0, 0, 0, 1], n),
        rng.choice([80, 443, 443, 443, 53, 22, 25, 993], n), rng.uniform(0.05, 0.8, n),
    ])


def _dos(rng: np.random.Generator, n: int) -> np.ndarray:
    duration = rng.uniform(0.5, 5, n)
    packets = rng.integers(3_000, 60_000, n)
    src_bytes = packets * rng.normal(70, 15, n).clip(40)
    return np.column_stack([
        duration, src_bytes, rng.uniform(0, 2_000, n), packets, packets / duration,
        src_bytes / packets, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.2, n),
        rng.integers(1, 3, n), rng.integers(1, 2, n), np.zeros(n),
        rng.choice([80, 443], n), rng.uniform(0.0001, 0.005, n),
    ])


def _portscan(rng: np.random.Generator, n: int) -> np.ndarray:
    packets = rng.integers(1, 4, n)
    return np.column_stack([
        rng.uniform(0.001, 0.05, n), packets * 60.0, rng.uniform(0, 60, n), packets,
        packets / rng.uniform(0.001, 0.05, n), rng.normal(58, 4, n), rng.uniform(0.8, 1.0, n),
        rng.uniform(0.4, 1.0, n), rng.integers(100, 5_000, n), rng.integers(1, 30, n), np.zeros(n),
        rng.integers(1, 65_535, n), rng.uniform(0.0001, 0.01, n),
    ])


def _bruteforce(rng: np.random.Generator, n: int) -> np.ndarray:
    duration = rng.uniform(1, 8, n)
    packets = rng.integers(15, 60, n)
    src_bytes = packets * rng.normal(110, 20, n).clip(60)
    return np.column_stack([
        duration, src_bytes, src_bytes * rng.uniform(0.8, 1.5, n), packets, packets / duration,
        src_bytes * 2 / packets, rng.uniform(0.05, 0.15, n), rng.uniform(0.0, 0.1, n),
        rng.integers(1, 2, n), rng.integers(1, 2, n), rng.integers(8, 200, n),
        rng.choice([22, 3389, 21, 23, 445], n), rng.uniform(0.01, 0.2, n),
    ])


def _exfiltration(rng: np.random.Generator, n: int) -> np.ndarray:
    duration = rng.uniform(30, 900, n)
    packets = rng.integers(5_000, 80_000, n)
    src_bytes = packets * rng.normal(1_300, 100, n).clip(800)
    return np.column_stack([
        duration, src_bytes, rng.uniform(1_000, 50_000, n), packets, packets / duration,
        src_bytes / packets, rng.uniform(0.0, 0.01, n), rng.uniform(0.0, 0.01, n),
        np.ones(n), np.ones(n), np.zeros(n),
        rng.choice([443, 8443, 53, 21], n), rng.uniform(0.001, 0.05, n),
    ])


def _botnet(rng: np.random.Generator, n: int) -> np.ndarray:
    duration = rng.uniform(0.1, 1.5, n)
    packets = rng.integers(4, 12, n)
    src_bytes = packets * rng.normal(180, 10, n)
    return np.column_stack([
        duration, src_bytes, src_bytes * rng.uniform(0.4, 0.7, n), packets, packets / duration,
        src_bytes * 1.6 / packets, rng.uniform(0.1, 0.25, n), rng.uniform(0.0, 0.05, n),
        np.ones(n), np.ones(n), np.zeros(n),
        rng.choice([6667, 8080, 4444, 1337, 443], n), rng.uniform(0.0, 0.002, n),
    ])


def _blend(rng: np.random.Generator, n: int, main, variant, share: float) -> np.ndarray:
    """Mix a share of hard 'variant' flows into a class so the boundaries overlap like real traffic."""
    k = int(round(n * share))
    parts = [main(rng, n - k)] + ([variant(rng, k)] if k else [])
    return np.vstack(parts)


def _benign_backup(rng: np.random.Generator, n: int) -> np.ndarray:
    """Legitimate large uploads (backups, cloud sync): look like exfiltration."""
    x = _exfiltration(rng, n)
    x[:, 1] *= rng.uniform(0.05, 0.6, n)            # smaller than typical exfil
    x[:, 2] *= rng.uniform(1.0, 20.0, n)            # chattier server responses
    x[:, 6] = rng.uniform(0.0, 0.05, n)
    x[:, 9] = rng.integers(1, 3, n)
    x[:, 12] = rng.uniform(0.01, 0.3, n)
    return x


def _benign_healthcheck(rng: np.random.Generator, n: int) -> np.ndarray:
    """Monitoring probes: tiny periodic flows, similar to C2 beacons."""
    x = _botnet(rng, n)
    x[:, 11] = rng.choice([443, 8080, 9100, 80], n)
    x[:, 12] = rng.uniform(0.0, 0.02, n)
    return x


def _benign_typos(rng: np.random.Generator, n: int) -> np.ndarray:
    """Users mistyping passwords: a few failed logins on SSH/RDP."""
    x = _bruteforce(rng, n)
    x[:, 10] = rng.integers(1, 6, n)
    x[:, 3] = rng.integers(10, 40, n)
    return x


def _stealth_scan(rng: np.random.Generator, n: int) -> np.ndarray:
    x = _portscan(rng, n)
    x[:, 8] = rng.integers(5, 60, n)                # few ports, slow
    x[:, 0] = rng.uniform(0.05, 2.0, n)
    x[:, 4] = x[:, 3] / x[:, 0]
    return x


def _slow_bruteforce(rng: np.random.Generator, n: int) -> np.ndarray:
    x = _bruteforce(rng, n)
    x[:, 10] = rng.integers(3, 12, n)
    x[:, 0] = rng.uniform(5, 60, n)
    x[:, 4] = x[:, 3] / x[:, 0]
    return x


def _low_rate_dos(rng: np.random.Generator, n: int) -> np.ndarray:
    x = _dos(rng, n)
    x[:, 3] = rng.integers(300, 3_000, n)
    x[:, 1] = x[:, 3] * rng.normal(90, 30, n).clip(40)
    x[:, 4] = x[:, 3] / x[:, 0]
    x[:, 5] = x[:, 1] / x[:, 3]
    x[:, 6] = rng.uniform(0.2, 0.7, n)
    return x


def _benign_mixed(rng: np.random.Generator, n: int) -> np.ndarray:
    k1, k2, k3 = int(n * 0.04), int(n * 0.03), int(n * 0.03)
    parts = [_benign(rng, n - k1 - k2 - k3)]
    for fn, k in ((_benign_backup, k1), (_benign_healthcheck, k2), (_benign_typos, k3)):
        if k:
            parts.append(fn(rng, k))
    return np.vstack(parts)


GENERATORS = {
    "benign": _benign_mixed,
    "dos": lambda r, n: _blend(r, n, _dos, _low_rate_dos, 0.25),
    "portscan": lambda r, n: _blend(r, n, _portscan, _stealth_scan, 0.30),
    "bruteforce": lambda r, n: _blend(r, n, _bruteforce, _slow_bruteforce, 0.30),
    "exfiltration": _exfiltration,
    "botnet": _botnet,
}


def generate(n: int = 20_000, mix: dict[str, float] | None = None, seed: int = 7):
    """Return (X, y) with ``n`` flows drawn according to ``mix``."""
    mix = mix or DEFAULT_MIX
    unknown = set(mix) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown classes: {unknown}")
    rng = np.random.default_rng(seed)
    total = sum(mix.values())
    xs, ys = [], []
    for label, share in mix.items():
        k = max(1, int(round(n * share / total)))
        xs.append(GENERATORS[label](rng, k))
        ys += [label] * k
    x = np.vstack(xs)
    y = np.asarray(ys)
    order = rng.permutation(len(y))
    return x[order], y[order]


def write_csv(path: str | Path, x: np.ndarray, y: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FEATURES + ["label"])
        for row, label in zip(x, y):
            w.writerow([f"{v:.6g}" for v in row] + [label])


def read_csv(path: str | Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    x = np.asarray([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.asarray([r["label"] for r in rows])
    return x, y
