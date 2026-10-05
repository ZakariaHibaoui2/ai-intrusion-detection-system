import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentinel_ids.cli import main
from sentinel_ids.features import FEATURES, LABELS, to_matrix
from sentinel_ids.model import HybridIDS, evaluate
from sentinel_ids.traffic import DEFAULT_MIX, generate, read_csv, write_csv


@pytest.fixture(scope="module")
def data():
    return generate(6_000, seed=1)


@pytest.fixture(scope="module")
def model(data):
    x, y = data
    return HybridIDS(n_estimators=80).fit(x, y)


def test_generator_shape_and_classes(data):
    x, y = data
    assert x.shape[1] == len(FEATURES)
    assert set(y) == set(LABELS)
    assert np.isfinite(x).all() and (x >= 0).all()


def test_generator_is_deterministic():
    a, _ = generate(500, seed=3)
    b, _ = generate(500, seed=3)
    assert np.array_equal(a, b)


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        generate(100, mix={"teleport": 1.0})


def test_holdout_metrics(data):
    r = evaluate(*data)
    assert r["macro_f1"] > 0.95
    assert r["detection_rate"] > 0.97
    assert r["false_positive_rate"] < 0.05


def test_zero_day_detection():
    """An attack family absent from training must still be flagged by the anomaly detector."""
    mix = {k: v for k, v in DEFAULT_MIX.items() if k != "exfiltration"}
    x_tr, y_tr = generate(6_000, mix=mix, seed=2)
    m = HybridIDS(n_estimators=80).fit(x_tr, y_tr)
    x_new, _ = generate(500, mix={"exfiltration": 1.0}, seed=9)
    x_ben, _ = generate(500, mix={"benign": 1.0}, seed=10)
    caught = np.mean([v.alert for v in m.predict(x_new)])
    fp = np.mean([v.alert for v in m.predict(x_ben)])
    assert caught > 0.7
    assert fp < 0.05


def test_save_load_roundtrip(tmp_path, model, data):
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = HybridIDS.load(path)
    x, _ = data
    assert [v.label for v in loaded.predict(x[:50])] == [v.label for v in model.predict(x[:50])]


def test_to_matrix_validates():
    with pytest.raises(ValueError):
        to_matrix([{"duration": 1}])


def test_csv_roundtrip(tmp_path):
    x, y = generate(200, seed=4)
    write_csv(tmp_path / "f.csv", x, y)
    x2, y2 = read_csv(tmp_path / "f.csv")
    assert x2.shape == x.shape and list(y2) == list(y)


def test_cli_train_and_detect(tmp_path, capsys):
    csv_path, model_path = tmp_path / "d.csv", tmp_path / "m.joblib"
    main(["generate", str(csv_path), "-n", "3000"])
    main(["train", "--data", str(csv_path), "-o", str(model_path)])
    main(["detect", str(csv_path), "-m", str(model_path)])
    out = capsys.readouterr()
    assert "alert(s)" in out.err


def test_api():
    from sentinel_ids.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    x, y = generate(50, mix={"portscan": 1.0}, seed=5)
    flow = dict(zip(FEATURES, map(float, x[0])))
    r = client.post("/score", json=flow).json()
    assert r["alert"] is True and r["label"] == "portscan"
    batch = client.post("/score/batch", json=[dict(zip(FEATURES, map(float, row))) for row in x[:10]])
    assert batch.json()["alerts"] == 10
    bad = dict(flow, syn_ratio=3)
    assert client.post("/score", json=bad).status_code == 422
