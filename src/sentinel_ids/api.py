"""REST API: score flows in real time.

Run:  SENTINEL_MODEL=model.joblib uvicorn sentinel_ids.api:app
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .features import FEATURES, to_matrix
from .model import HybridIDS
from .traffic import generate


class Flow(BaseModel):
    duration: float = Field(ge=0)
    src_bytes: float = Field(ge=0)
    dst_bytes: float = Field(ge=0)
    packets: float = Field(ge=0)
    pkt_rate: float = Field(ge=0)
    mean_pkt_size: float = Field(ge=0)
    syn_ratio: float = Field(ge=0, le=1)
    rst_ratio: float = Field(ge=0, le=1)
    distinct_dst_ports: float = Field(ge=0)
    distinct_dst_hosts: float = Field(ge=0)
    failed_logins: float = Field(ge=0)
    dst_port: float = Field(ge=0, le=65535)
    interarrival_std: float = Field(ge=0)


def _load_model() -> HybridIDS:
    path = os.environ.get("SENTINEL_MODEL")
    if path and os.path.exists(path):
        return HybridIDS.load(path)
    x, y = generate(8_000)  # demo model so the API works out of the box
    return HybridIDS(n_estimators=100).fit(x, y)


app = FastAPI(title="Sentinel IDS", version=__version__,
              description="Hybrid ML intrusion detection for network flows.")
MODEL = _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info() -> dict:
    return {
        "features": FEATURES,
        "classes": [str(c) for c in MODEL.clf.classes_],
        "top_features": MODEL.feature_importance()[:5],
    }


@app.post("/score")
def score(flow: Flow) -> dict:
    return MODEL.predict(to_matrix([flow.model_dump()]))[0].as_dict()


@app.post("/score/batch")
def score_batch(flows: list[Flow]) -> dict:
    if not flows:
        raise HTTPException(400, "empty batch")
    if len(flows) > 10_000:
        raise HTTPException(413, "batch too large (max 10000)")
    verdicts = MODEL.predict(to_matrix([f.model_dump() for f in flows]))
    return {"alerts": sum(v.alert for v in verdicts), "results": [v.as_dict() for v in verdicts]}
