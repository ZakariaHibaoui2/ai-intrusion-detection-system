"""Command line: generate data, train, evaluate, detect, serve."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .features import LABELS
from .model import HybridIDS, evaluate
from .traffic import DEFAULT_MIX, generate, read_csv, write_csv


def cmd_generate(a: argparse.Namespace) -> None:
    x, y = generate(a.n, seed=a.seed)
    write_csv(a.output, x, y)
    print(f"wrote {len(y)} flows to {a.output}")


def cmd_train(a: argparse.Namespace) -> None:
    x, y = read_csv(a.data) if a.data else generate(a.n, seed=a.seed)
    model = HybridIDS().fit(x, y)
    model.save(a.output)
    print(f"model trained on {len(y)} flows -> {a.output}")
    for name, imp in model.feature_importance()[:5]:
        print(f"  {name:<20} {imp:.3f}")


def cmd_evaluate(a: argparse.Namespace) -> None:
    x, y = read_csv(a.data) if a.data else generate(a.n, seed=a.seed)
    r = evaluate(x, y)
    print(r["report"])
    print(f"macro F1            : {r['macro_f1']:.4f}")
    print(f"attack detection    : {r['detection_rate']:.2%}")
    print(f"false positive rate : {r['false_positive_rate']:.2%}")


def cmd_zero_day(a: argparse.Namespace) -> None:
    """Hold one attack family out of training and measure how often it is still caught."""
    held = a.holdout
    mix = {k: v for k, v in DEFAULT_MIX.items() if k != held}
    x_tr, y_tr = generate(a.n, mix=mix, seed=a.seed)
    model = HybridIDS().fit(x_tr, y_tr)
    x_new, _ = generate(2_000, mix={held: 1.0}, seed=a.seed + 1)
    x_ben, _ = generate(2_000, mix={"benign": 1.0}, seed=a.seed + 2)
    caught = np.mean([v.alert for v in model.predict(x_new)])
    fp = np.mean([v.alert for v in model.predict(x_ben)])
    print(f"unseen '{held}' attacks detected : {caught:.2%}")
    print(f"benign false positives          : {fp:.2%}")


def cmd_detect(a: argparse.Namespace) -> None:
    model = HybridIDS.load(a.model)
    x, _ = read_csv(a.data)
    alerts = 0
    for i, v in enumerate(model.predict(x)):
        if v.alert:
            alerts += 1
            print(json.dumps({"flow": i, **v.as_dict()}))
    print(f"{alerts} alert(s) in {len(x)} flows", file=sys.stderr)


def cmd_serve(a: argparse.Namespace) -> None:
    import os

    import uvicorn

    if a.model:
        os.environ["SENTINEL_MODEL"] = a.model
    uvicorn.run("sentinel_ids.api:app", host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sentinel-ids", description="Hybrid AI intrusion detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create a labelled synthetic flow dataset")
    g.add_argument("output")
    g.add_argument("-n", type=int, default=20_000)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)

    for name, fn, helptext in (("train", cmd_train, "train and save a model"),
                               ("evaluate", cmd_evaluate, "hold-out evaluation report")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--data", help="CSV with feature columns + label (default: synthetic)")
        s.add_argument("-n", type=int, default=20_000)
        s.add_argument("--seed", type=int, default=7)
        if name == "train":
            s.add_argument("-o", "--output", default="model.joblib")
        s.set_defaults(func=fn)

    z = sub.add_parser("zero-day", help="measure detection of an attack family never seen in training")
    z.add_argument("--holdout", choices=[label for label in LABELS if label != "benign"], default="exfiltration")
    z.add_argument("-n", type=int, default=20_000)
    z.add_argument("--seed", type=int, default=7)
    z.set_defaults(func=cmd_zero_day)

    d = sub.add_parser("detect", help="score a CSV of flows and print alerts as JSON lines")
    d.add_argument("data")
    d.add_argument("-m", "--model", default="model.joblib")
    d.set_defaults(func=cmd_detect)

    s = sub.add_parser("serve", help="start the REST API")
    s.add_argument("-m", "--model")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
