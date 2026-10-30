"""Command line entry point: ``fraud-ml generate | train | drift | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from fraud_detection.artifact import ModelArtifact
from fraud_detection.drift import drift_table, render_markdown
from fraud_detection.features import build_features
from fraud_detection.generator import GeneratorConfig, generate_transactions
from fraud_detection.split import SplitConfig
from fraud_detection.train import TrainConfig, train, write_outputs


def _generate(args: argparse.Namespace) -> None:
    cfg = GeneratorConfig(
        n_cards=args.cards,
        n_merchants=args.merchants,
        days=args.days,
        start=args.start,
        seed=args.seed,
        amount_scale=args.amount_scale,
        fraud_multiplier=args.fraud_multiplier,
    )
    df = generate_transactions(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df):,} transactions ({df['is_fraud'].mean():.3%} fraud) to {args.out}")


def _train(args: argparse.Namespace) -> None:
    transactions = pd.read_parquet(args.data)
    result = train(transactions, TrainConfig(seed=args.seed, review_cost=args.review_cost))
    paths = write_outputs(result, args.artifacts, args.reports)
    test = result.report["models"][result.artifact.model_name]["test"]
    print(f"selected {result.artifact.model_name} -> {result.artifact.model_version}")
    print(json.dumps({k: round(v, 4) for k, v in test.items()}, indent=2))
    for path in paths:
        print(f"wrote {path}")


def _drift(args: argparse.Namespace) -> None:
    artifact = ModelArtifact.load(args.model)
    transactions = pd.read_parquet(args.data)
    features = build_features(transactions, artifact.feature_config)
    since = features["timestamp"].min() + pd.Timedelta(days=SplitConfig().burn_in_days)
    if args.since:
        since = max(since, pd.Timestamp(args.since))
    current = features.loc[features["timestamp"] >= since]
    if current.empty:
        raise SystemExit(f"no transactions at or after {since}")
    columns = {name: current[name].to_numpy() for name in artifact.feature_names}
    columns["model_score"] = artifact.score_frame(current)
    table = drift_table(artifact.reference_profiles, columns)
    context = {
        "model": f"{artifact.model_name} `{artifact.model_version}`",
        "reference": "training period of the model (profiles stored in the artefact)",
        "current": f"`{args.data}`, {len(current):,} transactions from {since:%Y-%m-%d %H:%M}",
    }
    markdown = render_markdown(table, "Drift report (PSI)", context)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)
    print(markdown)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from fraud_detection.api import create_app

    app = create_app(model_path=args.model, history_path=args.history)
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-ml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic transaction log (parquet)")
    g.add_argument("--out", type=Path, default=Path("data/transactions.parquet"))
    g.add_argument("--cards", type=int, default=GeneratorConfig.n_cards)
    g.add_argument("--merchants", type=int, default=GeneratorConfig.n_merchants)
    g.add_argument("--days", type=int, default=GeneratorConfig.days)
    g.add_argument("--start", default=GeneratorConfig.start)
    g.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    g.add_argument("--amount-scale", type=float, default=1.0, help="simulate price drift")
    g.add_argument("--fraud-multiplier", type=float, default=1.0, help="simulate a fraud wave")
    g.set_defaults(func=_generate)

    t = sub.add_parser("train", help="train, evaluate, write artefact + model card")
    t.add_argument("--data", type=Path, default=Path("data/transactions.parquet"))
    t.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    t.add_argument("--reports", type=Path, default=Path("reports"))
    t.add_argument("--seed", type=int, default=TrainConfig.seed)
    t.add_argument("--review-cost", type=float, default=TrainConfig.review_cost)
    t.set_defaults(func=_train)

    d = sub.add_parser("drift", help="PSI of a new batch against the model's training profile")
    d.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    d.add_argument("--data", type=Path, required=True)
    d.add_argument("--since", help="only compare transactions at or after this timestamp")
    d.add_argument("--out", type=Path, default=Path("reports/drift_report.md"))
    d.set_defaults(func=_drift)

    s = sub.add_parser("serve", help="run the FastAPI scoring service")
    s.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    s.add_argument("--history", type=Path, help="transaction log to warm up card state from")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
