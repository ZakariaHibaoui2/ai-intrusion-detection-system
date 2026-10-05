"""Hybrid detector: Random Forest for known attacks + Isolation Forest for novelties."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .features import FEATURES, log_transform


@dataclass
class Verdict:
    label: str            # most likely known class
    confidence: float     # classifier probability for that class
    anomaly_score: float  # > 0 means "unlike anything benign seen in training"
    alert: bool           # final decision
    reason: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class HybridIDS:
    """Two complementary detectors.

    * A **Random Forest** recognises attack families seen during training.
    * An **Isolation Forest** fitted on benign traffic only scores how unusual a flow is,
      so attacks that were never labelled (zero-days) still raise an alert.
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.02, seed: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced", n_jobs=-1, random_state=seed
        )
        self.iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
        self.anomaly_threshold = 0.0
        self.fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> HybridIDS:
        xt = log_transform(x)
        self.clf.fit(xt, y)
        benign = xt[y == "benign"]
        if len(benign) == 0:
            raise ValueError("training data must contain benign flows")
        self.iso.fit(benign)
        # threshold = 99th percentile of benign anomaly scores -> ~1% false-positive budget
        self.anomaly_threshold = float(np.percentile(-self.iso.score_samples(benign), 99))
        self.fitted = True
        return self

    def anomaly_scores(self, x: np.ndarray) -> np.ndarray:
        """Positive = more anomalous than 99% of benign training traffic."""
        return -self.iso.score_samples(log_transform(x)) - self.anomaly_threshold

    def predict(self, x: np.ndarray) -> list[Verdict]:
        if not self.fitted:
            raise RuntimeError("model is not trained")
        xt = log_transform(x)
        proba = self.clf.predict_proba(xt)
        classes = self.clf.classes_
        anomaly = self.anomaly_scores(x)
        out = []
        for p, a in zip(proba, anomaly):
            i = int(np.argmax(p))
            label, conf = str(classes[i]), float(p[i])
            if label != "benign":
                alert, reason = True, f"classified as {label}"
            elif a > 0:
                alert, reason = True, "benign-looking but anomalous: possible unknown attack"
            else:
                alert, reason = False, "normal"
            out.append(Verdict(label, round(conf, 4), round(float(a), 4), alert, reason))
        return out

    def predict_one(self, flow: dict) -> Verdict:
        return self.predict(np.asarray([[float(flow[f]) for f in FEATURES]]))[0]

    def feature_importance(self) -> list[tuple[str, float]]:
        imp = self.clf.feature_importances_
        return sorted(zip(FEATURES, map(float, imp)), key=lambda t: -t[1])

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> HybridIDS:
        model = joblib.load(path)
        if not isinstance(model, HybridIDS):
            raise TypeError("not a HybridIDS model file")
        return model


def evaluate(x: np.ndarray, y: np.ndarray, test_size: float = 0.25, seed: int = 42) -> dict:
    """Train on a stratified split and report metrics on the held-out part."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, stratify=y, random_state=seed)
    model = HybridIDS(seed=seed).fit(x_tr, y_tr)
    verdicts = model.predict(x_te)
    pred = np.asarray([v.label for v in verdicts])
    alerts = np.asarray([v.alert for v in verdicts])
    is_attack = y_te != "benign"
    labels = sorted(set(y))
    return {
        "model": model,
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "detection_rate": float(alerts[is_attack].mean()),
        "false_positive_rate": float(alerts[~is_attack].mean()),
        "report": classification_report(y_te, pred, digits=3, zero_division=0),
        "confusion": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "labels": labels,
    }
