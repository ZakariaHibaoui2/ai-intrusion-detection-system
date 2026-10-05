# Sentinel IDS — Hybrid AI Network Intrusion Detection

> Detects network attacks in flow data with two complementary models: a **Random Forest** that recognises known attack families, and an **Isolation Forest** trained only on normal traffic that flags never-seen-before (zero-day) behaviour. It ships with a CLI, a REST API, and a Docker image.

[![CI](https://github.com/ZakariaHibaoui2/ai-intrusion-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/ZakariaHibaoui2/ai-intrusion-detection-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
 flow features (13)                     ┌──────────────────────────────┐
 duration, bytes, packets, rates,  ───► │ log1p transform              │
 SYN/RST ratios, fan-out (ports,        └──────────────┬───────────────┘
 hosts), failed logins, port, jitter                   │
                              ┌────────────────────────┴────────────────────────┐
                              ▼                                                 ▼
             Random Forest (200 trees, balanced)              Isolation Forest (benign only)
             benign / dos / portscan / bruteforce /           anomaly score, threshold =
             exfiltration / botnet                            99th pct of benign → ~1% FP budget
                              │                                                 │
                              └──────────────► decision ◄───────────────────────┘
                     attack class → ALERT · benign but anomalous → ALERT (possible zero-day) · else OK
```

## Results

All numbers below were measured with the commands in this README, on synthetic data (see [Data](#data)).

**Known attacks** (`sentinel-ids evaluate -n 30000`, 25% hold-out):

| Metric | Value |
|---|---|
| Macro F1 (6 classes) | **0.995** |
| Attack detection rate | **100%** |
| False-positive rate on benign flows | **1.47%** |

**Zero-day experiment** (`sentinel-ids zero-day --holdout <family>`): one attack family is removed from training entirely, then we measure how often it is still caught.

| Unseen family | Detected | Benign FP |
|---|---|---|
| DoS | 99.5% | 1.45% |
| Port scan | 100% | 1.50% |
| Exfiltration | 82.0% | 1.70% |
| Brute force | 10.8% | 1.45% |
| Botnet beacon | 0.2% | 1.45% |

**Takeaway:** volumetric and fan-out attacks stand out from normal traffic, so the anomaly model catches them unseen. Low-and-slow brute force and C2 beacons deliberately **mimic benign patterns** (password typos, monitoring probes). Point-in-time flow features can't separate them, which is why real SOCs add longer time-window and per-host baselining features. This is the next step on the roadmap.

## Quick start

```bash
git clone https://github.com/ZakariaHibaoui2/ai-intrusion-detection-system.git
cd ai-intrusion-detection-system
pip install -e ".[dev]"

sentinel-ids generate flows.csv -n 20000          # labelled synthetic dataset
sentinel-ids evaluate --data flows.csv            # classification report + detection/FP rates
sentinel-ids zero-day --holdout exfiltration      # unseen-attack experiment
sentinel-ids train --data flows.csv -o model.joblib
sentinel-ids detect flows.csv -m model.joblib     # JSON-lines alerts
sentinel-ids serve -m model.joblib                # REST API on :8000
```

### REST API

```bash
curl -X POST localhost:8000/score -H "content-type: application/json" -d '{
  "duration":0.01,"src_bytes":60,"dst_bytes":0,"packets":1,"pkt_rate":100,"mean_pkt_size":60,
  "syn_ratio":1,"rst_ratio":1,"distinct_dst_ports":1200,"distinct_dst_hosts":3,
  "failed_logins":0,"dst_port":4455,"interarrival_std":0.001}'
# {"label":"portscan","confidence":1.0,"anomaly_score":0.0665,"alert":true,"reason":"classified as portscan"}
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/model` | Features, classes, top feature importances |
| POST | `/score` | Score one flow (validated with Pydantic) |
| POST | `/score/batch` | Score up to 10,000 flows |

Interactive docs: `http://localhost:8000/docs`.

### Docker

```bash
docker build -t sentinel-ids .
docker run -p 8000:8000 sentinel-ids
```

## Data

`sentinel_ids.traffic` generates labelled flows whose statistics follow each traffic class, including **hard cases**:

- benign backups that look like exfiltration, monitoring probes that look like beacons, and users mistyping passwords
- stealth port scans, low-and-slow brute force, and low-rate DoS

This keeps the project self-contained and reproducible. To train on real traffic, export flows (e.g. CIC-IDS2017 / Zeek / NetFlow) into a CSV with the same 13 feature columns plus `label` and pass `--data`.

## Project structure

```
src/sentinel_ids/
├── features.py   # feature schema, validation, log transform
├── traffic.py    # synthetic flow generator (6 classes + hard cases), CSV I/O
├── model.py      # HybridIDS (RF + IsolationForest), evaluation
├── api.py        # FastAPI service
└── cli.py        # generate / train / evaluate / zero-day / detect / serve
tests/            # 10 tests: data, metrics, zero-day, persistence, CLI, API
```

## Development

```bash
ruff check .
pytest -q
```

## License

[MIT](LICENSE)
