"""Flow feature schema shared by the generator, the model and the API."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

FEATURES: list[str] = [
    "duration",            # seconds
    "src_bytes",           # bytes client -> server
    "dst_bytes",           # bytes server -> client
    "packets",             # total packets
    "pkt_rate",            # packets / second
    "mean_pkt_size",       # bytes / packet
    "syn_ratio",           # SYN packets / packets
    "rst_ratio",           # RST packets / packets
    "distinct_dst_ports",  # ports contacted by the source in the window
    "distinct_dst_hosts",  # hosts contacted by the source in the window
    "failed_logins",       # authentication failures in the window
    "dst_port",            # destination port of the flow
    "interarrival_std",    # jitter of packet inter-arrival times (s)
]

LABELS: list[str] = ["benign", "dos", "portscan", "bruteforce", "exfiltration", "botnet"]


def to_matrix(flows: Iterable[Mapping[str, float]]) -> np.ndarray:
    """Convert flow dicts to an (n, len(FEATURES)) float matrix, validating keys."""
    rows = []
    for i, flow in enumerate(flows):
        missing = [f for f in FEATURES if f not in flow]
        if missing:
            raise ValueError(f"flow {i} is missing features: {missing}")
        rows.append([float(flow[f]) for f in FEATURES])
    return np.asarray(rows, dtype=float).reshape(-1, len(FEATURES))


def log_transform(x: np.ndarray) -> np.ndarray:
    """Heavy-tailed traffic counters behave better on a log scale."""
    return np.log1p(np.clip(x, 0, None))
